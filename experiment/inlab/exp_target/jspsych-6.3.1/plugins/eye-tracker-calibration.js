/*
 * Plugin for drawing dots on a canvas
 * By Clara Kuper, 2022
 */

jsPsych.plugins["eye-tracker-calibration"] = (function () {

    let plugin = {};

    plugin.info = {
        name: "eye-tracker-calibration",
        parameters: {
            fixation_target_color: {
                type: jsPsych.plugins.parameterType.STRING, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
                default: '#FFFFFF',
                description: "The color of the dot participants should look at."
            },
            fixation_target_positions_x: {
                type: jsPsych.plugins.parameterType.INT,
                default: undefined,
                description: "The x positions of the fixation targets"
            },
            fixation_target_positions_y: {
                type: jsPsych.plugins.parameterType.INT,
                default: undefined,
                description: "The y position of the fixation targets"
            },
            shrink_duration: {
                type: jsPsych.plugins.parameterType.INT,
                default: 2000,
                description: 'How long the fixation dots take to disappear [ms]'
            },
            wait_after: {
                type: jsPsych.plugins.parameterType.INT,
                default: 500,
                description: "waiting time before we close the trial"
            },
            background_color: {
                type: jsPsych.plugins.parameterType.STRING,
                default: '#666666',
                description: "default color of the background"
            },
            pixel_per_degree: {
                type:jsPsych.plugins.parameterType.INT,
                default: 3.1415926535 * 10,
                description: 'How many pixels represent one degree visual angle'
            },
            tag_size: {
                type:jsPsych.plugins.parameterType.INT,
                default: 15,
                description: 'How many pixels each tile of the april tag should have.'
            },
            photodiode_color: {
                type:jsPsych.plugins.parameterType.STRING,
                default: '#000000',
                description: 'The color to which the photodiode changes'
            },
            black: {
                type:jsPsych.plugins.parameterType.STRING,
                default: '#000000',
                description: 'dark values'
            },
            white: {
                type:jsPsych.plugins.parameterType.STRING,
                default: '#FFFFFF',
                description: 'light values'
            },
        },
    };

    plugin.trial = function (display_element, trial) {

        // timing stuff
        let change_times = [];
        let photodiode_colors = [];
        let start_trial = true;

        // intialize parameters
        let photodiode_color = trial.background_color
        let n_circles = 0;

        let fixation_size = trial.pixel_per_degree;
        let shrink_each = trial.shrink_duration/fixation_size;

        let time_since_shrink;
        let time_since_switch;
        let write_switch_time = false;

        let targets_to_present = []
        for (let i = 0; i<trial.fixation_target_positions_x.length; i++){
            targets_to_present.push(i)
        }

        let get_next_target = function(array){
            let index = Math.floor(Math.random()*array.length);
            let next_target = array[index];
            array.splice(index, 1);

            return next_target
        }

        let next_target = get_next_target(targets_to_present);

        // create a canvas   element
        display_element.innerHTML = '<canvas></canvas>';

        // initialize the canvas
        let canvas = document.querySelector('canvas');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        // get canvas context
        let c = canvas.getContext('2d');

        // initialize an animation request handler
        let AnimationRequest;

        let perform_updates = function (photodiode_square, photodiode_color, touch_circle, touch_circle_size) {
            photodiode_square.change_color(photodiode_color);
            touch_circle.change_radius(touch_circle_size);
        };

        let get_current_position_and_color = function (timestamp){
            let new_circle_rad = touch_circle.rad;

            if (new_circle_rad <= 1){
                new_circle_rad = trial.pixel_per_degree;
                n_circles += 1;
                // check if trial is over
                if (n_circles > trial.fixation_target_positions_x.length){
                    end_trial();
                } else {
                    next_target = get_next_target(targets_to_present);
                    touch_circle.change_position(
                        trial.fixation_target_positions_x[next_target],
                        trial.fixation_target_positions_y[next_target]
                    )
                }
            } else if (timestamp-time_since_shrink >= shrink_each){
                time_since_shrink = timestamp
                new_circle_rad -= 1
            }

            write_switch_time = false;
            if (timestamp - time_since_switch > 1000) {
                write_switch_time = true;
                photodiode_color = [trial.white, trial.black][+(photodiode_color === trial.white)];
                time_since_switch = timestamp;
            }
            return [photodiode_color, new_circle_rad];
        };

        let animate_setup = function (timestamp){
            if (!start_trial){
                requestAnimationFrame(animate_setup)
            } else {
                time_since_switch = timestamp;
                time_since_shrink = timestamp;
                photodiode_color = trial.white;
                requestAnimationFrame(animate);
            }
            c.clearRect(0, 0, innerWidth, innerHeight);
            background_rect.draw();
            photodiode_check.draw();
        }


        let animate = function (timestamp) {
            AnimationRequest = requestAnimationFrame(animate);
            let repaint_values = get_current_position_and_color(timestamp);
            perform_updates(photodiode_check, repaint_values[0], touch_circle, repaint_values[1]);
            c.clearRect(0, 0, innerWidth, innerHeight);
            background_rect.draw();
            photodiode_check.draw();
            touch_circle.draw();

            if (photodiode_color === trial.white){
                for (let i=0; i< apriltag_1.length; i++){
                    apriltag_1[i].draw();
                    apriltag_2[i].draw();
                    apriltag_3[i].draw();
                }
            }
            if (write_switch_time){
                change_times.push(timestamp);
            }
        };

        let end_trial = function () {
            // break request animation frame loop
            cancelAnimationFrame(AnimationRequest);

            // kill any remaining setTimeout handlers
            jsPsych.pluginAPI.clearAllTimeouts();
            // gather the data to store for the trial
            let trial_data = {
                //timing
                "change_timestamps": change_times,
                "presented_colors": photodiode_colors,
                "windowWidth": window.innerWidth, // the width of the screen
                "windowHeight": window.innerHeight, // the height of the screen
                "userInfo": navigator.userAgent, // some information about the user device
                "platform": navigator.platform, // some information about the web browser
            };
            // clear the display
            display_element.innerHTML = '';
            // reset theme
            head.removeChild(head.lastElementChild);
            // move on to the next trial after the waiting time
            setTimeout(jsPsych.finishTrial, trial.wait_after, trial_data);
        };

        // draw the square in the center
        let background_rect = new Rectangle(0, 0, innerWidth, innerHeight, trial.background_color, c);

        let touch_circle = new Circle(
            trial.fixation_target_positions_x[next_target],
            undefined,
            trial.fixation_target_positions_y[next_target],
            undefined,
            trial.pixel_per_degree,
            trial.fixation_target_color,
            undefined,
            c
        );

        // draw the rectangle for the photodiode
        let photodiode_check = new Rectangle(innerWidth-40, 0,
            40, 20, photodiode_color, c);
        let apriltag_1 = april_tag_drawer(tag_id_1, trial.tag_size, c, 0, 0);
        let apriltag_2 = april_tag_drawer(tag_id_2,
            trial.tag_size,
            c,
            innerHeight - (tag_id_2.length * trial.tag_size),
            0)
        let apriltag_3 = april_tag_drawer(tag_id_3,
            trial.tag_size,
            c,
            innerHeight - (tag_id_2.length * trial.tag_size),
            window.innerWidth - (tag_id_2.length * trial.tag_size));

        // load new stylesheet and start animating
        let href = "custom_styles.css"
        let head = document.getElementsByTagName('head')[0];
        let trialStyleSheet = document.createElement('link');

        trialStyleSheet.type = "text/css";
        trialStyleSheet.rel = "stylesheet";
        trialStyleSheet.href = href;

        trialStyleSheet.onload = function(){console.log('style sheet was loaded');
            requestAnimationFrame(animate_setup)
        }
        trialStyleSheet.onerror = function(){console.log('The style sheet for the trial could not load!')};
        head.appendChild(trialStyleSheet)
    };

    return plugin;
})();
