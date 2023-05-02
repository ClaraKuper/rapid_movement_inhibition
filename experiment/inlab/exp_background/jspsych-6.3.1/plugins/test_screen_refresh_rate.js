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

jsPsych.plugins["screen-refresh-test"] = (function () {

    let plugin = {};

    plugin.info = {
        // every parameter that we need to pass to the function should be specified here.
        // we can either set a default value
        // or we can set it to undefined
        name: 'screen-refresh-test',
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
                description: 'When to move on to the next trial.'
            },
        },
    };

    plugin.trial = function (display_element, trial) {

        let trialDur = trial.trialDuration;
        let waitAfter = trial.waitAfter;
        let timestamp_list = [];
        let animation_request;
        let now_time;
        let rate;

        let animate_screen = function (timestamp) {
            animation_request = requestAnimationFrame(animate_screen)
            if (now_time == null){
                now_time = timestamp
            }
            timestamp_list.push(timestamp - now_time);
            now_time = timestamp;
        }

        let end_trial = function () {
            jsPsych.pluginAPI.clearAllTimeouts();
            cancelAnimationFrame(animation_request);
            // get the median rate
            let unsorted_timestamps = timestamp_list;
            console.log(unsorted_timestamps);
            let sorted_timestamps = timestamp_list.sort((a,b) => a - b)
            let mid = Math.floor(timestamp_list.length/2)

            if (timestamp_list.length % 2 !== 0){
                rate = sorted_timestamps[mid]
            } else {
                rate = (sorted_timestamps[mid-1] + sorted_timestamps[mid])/2
            }


            let trial_data = {
                "refresh_rate": rate, // list of all animation frames
                "timestamps": unsorted_timestamps,
                "windowWidth": window.innerWidth, // the width of the screen
                "windowHeight": window.innerHeight, // the height of the screen
            };

            // move on to the next trial after the waiting time
            setTimeout(jsPsych.finishTrial, waitAfter, trial_data);
        };

        requestAnimationFrame(animate_screen)
        setTimeout(end_trial, trialDur)

    };
    return plugin;
})();
  