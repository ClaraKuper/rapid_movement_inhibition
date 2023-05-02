/**
 * custom-refresh-estimate
 * Josh de Leeuw
 * edited by Clara Kuper 2022
 *
 * plugin for displaying a canvas and measuring the time between animation frames
 *
 * documentation: docs.jspsych.org
 *
 **/

jsPsych.plugins["refresh-estimate"] = (function () {

    let plugin = {};

    plugin.info = {
        // every parameter that we need to pass to the function should be specified here.
        // we can either set a default value
        // or we can set it to undefined
        name: 'refresh-estimate',
        description: '',
        parameters: {
            // screen element parameters
            // timing parameters
            trialDuration: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Trial duration',
                default: 1000,
                description: 'How long to show the trial.'
            },
            waitAfter: {
                type: jsPsych.plugins.parameterType.INT,
                pretty_name: 'Wait after finish',
                default: 500,
                description: 'Message to display during screen refresh.'
            },
        },
    };

    plugin.trial = function (display_element, trial) {

        // copy some values from trial
        // this is not necessary, but helps to keep an overview
        let trialDur = trial.trialDuration;
        let waitAfter = trial.waitAfter
        let inTrial = true;

        // canvas elements
        let empty_canvas = '<canvas></canvas>';
        display_element.innerHTML = empty_canvas;

        // initialize canvas
        let canvas = document.querySelector('canvas');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        // get canvas context
        let c = canvas.getContext('2d');

        // PARAMETERS
        // variables to track
        // timing
        let AnimationFrameOn = [];

        // draw objects on the canvas
        // upper and lower rectangle
        function Circle(x, y, rad, color) {

            // initiate properties
            this.x = x; // x position
            this.y = y; // y position
            this.rad = rad; // initial radius
            this.color = color;

            this.draw = function(){
                // start drawing a new circle
                c.beginPath();
                c.arc(this.x, this.y, this.rad, 0, Math.PI * 2, false)
                c.fillStyle = this.color;
                c.fill();
            };

            this.update = function(){
                // change the radius
                this.rad += 0.1
                this.draw()
            }
        }

        let stimulus_circle = new Circle(window.innerWidth/2, window.innerHeight/2, 1, '#2274A5');

        // define the animation
        let animate = function(e){
            if (inTrial){
                requestAnimationFrame(animate);
                AnimationFrameOn.push(jsPsych.totalTime());
                c.clearRect(0,0, innerWidth, innerHeight);
                stimulus_circle.update();
            }
        }

        // function to end trial when it is time
        let end_trial = function() {

            inTrial = false;
            // kill any remaining setTimeout handlers
            jsPsych.pluginAPI.clearAllTimeouts();

            // print the animation frames in the console
            console.log(AnimationFrameOn);

            // compute the duration between Animation Frames
            let new_Array = [];
            for (let i = 1; i < AnimationFrameOn.length; i++) new_Array.push(AnimationFrameOn[i] - AnimationFrameOn[i - 1]);

            console.log(new_Array);


            // gather the data to store for the trial
            let trial_data = {
                //timing
                "AnimationFrames": AnimationFrameOn, // list of all animation frames
                "windowWidth": window.innerWidth, // the width of the screen
                "windowHeight": window.innerHeight, // the height of the screen
            };

            // clear the display
            c.clearRect(0,0, innerWidth, innerHeight);
            // move on to the next trial after the waiting time
            setTimeout(jsPsych.finishTrial(trial_data), waitAfter);
        };


        // start the animation
        animate();
        // set a timeout to end the trial if it lasts too long
        setTimeout(end_trial, trialDur);
    };
    return plugin;
})();
  