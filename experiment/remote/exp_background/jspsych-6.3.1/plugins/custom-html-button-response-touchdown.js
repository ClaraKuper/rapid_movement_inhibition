/**
 * jspsych-html-button-response-touchdown
 * Josh de Leeuw
 *
 * plugin for displaying a stimulus and getting a touch response
 * edited by Clara Kuper 2021
 * edited version works for touch responses and includes time stamps for touch events
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["html-button-response-touchdown"] = (function() {

  // define plugin info
  var plugin = {};

  plugin.info = {
    name: 'html-button-response-touchdown',
    description: '',
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.HTML_STRING,
        pretty_name: 'Stimulus',
        default: undefined,
        description: 'The HTML string to be displayed'
      },
      choices: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Choices',
        default: undefined,
        array: true,
        description: 'The labels for the buttons.'
      },
      button_html: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Button HTML',
        default: '<button class="jspsych-btn">%choice%</button>',
        array: true,
        description: 'The html of the button. Can create own style.'
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Prompt',
        default: null,
        description: 'Any content here will be displayed under the button.'
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Stimulus duration',
        default: null,
        description: 'How long to hide the stimulus.'
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: 'Trial duration',
        default: null,
        description: 'How long to show the trial.'
      },
      margin_vertical: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Margin vertical',
        default: '0px',
        description: 'The vertical margin of the button.'
      },
      margin_horizontal: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: 'Margin horizontal',
        default: '8px',
        description: 'The horizontal margin of the button.'
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: 'Response ends trial',
        default: true,
        description: 'If true, then trial will end when user responds.'
      },
    }
  }

  plugin.trial = function(display_element, trial) {

    // make the html info
    // display stimulus
    var html = '<div id="jspsych-html-button-response-stimulus">'+trial.stimulus+'</div>';

    //display buttons
    var buttons = [];
    if (Array.isArray(trial.button_html)) {
      if (trial.button_html.length == trial.choices.length) {
        buttons = trial.button_html;
      } else {
        console.error('Error in html-button-response plugin. The length of the button_html array does not equal the length of the choices array');
      }
    } else {
      for (var i = 0; i < trial.choices.length; i++) {
        buttons.push(trial.button_html);
      }
    }
    html += '<div id="jspsych-html-button-response-btngroup">';
    for (var i = 0; i < trial.choices.length; i++) {
      var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
      html += '<div class="jspsych-html-button-response-button" style="display: inline-block; margin:'+trial.margin_vertical+' '+trial.margin_horizontal+'" id="jspsych-html-button-response-button-' + i +'" data-choice="'+i+'">'+str+'</div>';
    }
    html += '</div>';

    // create flashes

    html += "<div id='flashup' style='position:absolute; left:0; top:0; width:100%; height:33%; background-color:rgba(255,255,255,0);'></div>";
    html += "<div id='flashdown' style='position:absolute; left:0; bottom:0; width:100%; height:33%; background-color:rgba(255,255,255,0);'></div>";


    //show prompt if there is one
    if (trial.prompt !== null) {
      html += trial.prompt;
    }
    display_element.innerHTML = html;

    // variables to track
    let js_lifttime;
    let stimRect;
    let allTouches = [];
    let targetTouched = false;
    let trialTimes = [];

    let flashOn;
    let flashOff;

    let xyCoords = {
        liftX: null,
        liftY: null,
        touchX: null,
        touchY: null,
    };

    let xyCrossScreen = {
      touchX: null,
      touchY: null,
    };

    // define flash time
    let flash_time = Math.random()*200;

    // record all touches across the document
    document.addEventListener('click', function(e){

        xyCrossScreen.touchX = e.pageX;
        xyCrossScreen.touchY = e.pageY;
        if (xyCrossScreen.touchX == null){
          xyCrossScreen.touchX = e.touches[0].pageX;
          xyCrossScreen.touchY = e.touches[0].pageY;
        };

        allTouches.push(xyCrossScreen.touchX, xyCrossScreen.touchY, jsPsych.totalTime());
    })

    // add event listeners to buttons
    for (var i = 0; i < trial.choices.length; i++) {
      display_element.querySelector('#jspsych-html-button-response-button-' + i).addEventListener('touchstart', onTouchStart);
      display_element.querySelector('#jspsych-html-button-response-button-' + i).addEventListener('touchend', onTouchEnd);
      //display_element.querySelector('#jspsych-html-button-response-button-' + i).addEventListener('mousedown', onTouchStart);
      //display_element.querySelector('#jspsych-html-button-response-button-' + i).addEventListener('mouseup', onTouchEnd);
    };

    function onTouchEnd(e){
      // handles lift events
      let lift_time = jsPsych.totalTime();
      js_lifttime = lift_time;
      xyCoords.liftX = xyCoords.touchX;
      xyCoords.liftY = xyCoords.touchY;
    }

    function onTouchStart(e){
        // handles touch events
        var choice = e.currentTarget.getAttribute('data-choice'); // don't use dataset for jsdom compatibility
        xyCoords.touchX = e.pageX;
        xyCoords.touchY = e.pageY;
        if (xyCoords.touchX == null){
          xyCoords.touchX = e.touches[0].pageX;
          xyCoords.touchY = e.touches[0].pageY;
        };

        after_response(choice);
        targetTouched = true;
      };


    // start time
    let start_time = jsPsych.totalTime();

    // set variables to track the appearance/disappearence of the flash
    let showflash = true;
    let hideflash = true;
    let in_trial = true;

    // trial run function
    let run_trial = function () {
          let time_passed = jsPsych.totalTime() - start_time;
          if (time_passed < trial.trial_duration && in_trial) {
              // this code runs till the time is over
              // check if it's time to show the flash
              if (time_passed >= flash_time && showflash) {
                if (trial.show_flash){
                  document.getElementById('flashup').style.backgroundColor = 'rgb(255,255,255)';
                  document.getElementById('flashdown').style.backgroundColor = 'rgb(255,255,255)';
                } else {
                  document.getElementById('flashup').style.backgroundColor = 'rgb(255,255,255,0)';
                  document.getElementById('flashdown').style.backgroundColor = 'rgb(255,255,255,0)';
                };

                flashOn = jsPsych.totalTime();
                showflash = false;

                // or time to remove the flash
              } else if (time_passed >= flash_time + trial.flash_dur && hideflash) {
                document.getElementById('flashup').style.backgroundColor = 'rgba(255,255,255,0)';
                document.getElementById('flashdown').style.backgroundColor = 'rgba(255,255,255,0)';

                flashOff = jsPsych.totalTime();
                hideflash = false;
              }
              ;

              // get a time series how often/when this function was called
              trialTimes.push(time_passed);

              // repeat this function after 5 ms
              setTimeout(run_trial, 5);
            }
      };

    // store response
    var response = {
      rt: null,
      button: null
    };

    // function to handle responses by the subject
    function after_response(choice) {

      // measure rt
      var end_time = jsPsych.totalTime();
      var rt = end_time - start_time;
      js_endtime = end_time;
      response.button = choice;
      response.rt = rt;

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      display_element.querySelector('#jspsych-html-button-response-stimulus').className += 'responded';

      // disable all the buttons after a response
      var btns = document.querySelectorAll('.jspsych-html-button-response-button button');
      for(var i=0; i<btns.length; i++){
        //btns[i].removeEventListener('touchstart', eventResponse);
        //btns[i].removeEventListener('touchend', handlelift);
        btns[i].setAttribute('disabled', 'disabled');
      }

      if (trial.response_ends_trial) {
        setTimeout(end_trial,100);
      }
    };

    // function to end trial when it is time
    function end_trial() {

      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // gather the data to store for the trial
      var trial_data = {
        "rt": response.rt,
        "stimulus": trial.stimulus,
        "button_pressed": response.button,
        "touchX": xyCoords.touchX,
        "touchY": xyCoords.touchY,
        "liftX": xyCoords.liftX,
        "liftY": xyCoords.liftY,
        // custom timing
        "js_start": start_time,
        "js_touchdown": js_endtime,
        "js_touchup": js_lifttime,
        "js_end": jsPsych.totalTime(),
        "xy_stim": stimRect,
        "button0-x": document.querySelector('#jspsych-html-button-response-button-0').getBoundingClientRect().x,
        "button0-y": document.querySelector('#jspsych-html-button-response-button-0').getBoundingClientRect().y,
        "all_touches": allTouches
      };

      // remove all event listeners

      // clear the display
      display_element.innerHTML = '';

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // hide image if timing is set
    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        display_element.querySelector('#jspsych-html-button-response-stimulus').style.visibility = 'hidden';
      }, trial.stimulus_duration);
    }

    // end trial if time limit is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function() {
        end_trial();
      }, trial.trial_duration);
    }

  run_trial();
  };

  return plugin;
})();
