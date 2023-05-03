/*
 * Plugin for checking the setup of the screen
 * By Clara Kuper, 2022
 */

jsPsych.plugins["canvas-setup-check"] = (function () {

    let plugin = {};

    plugin.info = {
        name: "canvas-setup-check",
        parameters: {
            white: {
                type: jsPsych.plugins.parameterType.STRING, // BOOL, STRING, INT, FLOAT, FUNCTION, KEYCODE, SELECT, HTML_STRING, IMAGE, AUDIO, VIDEO, OBJECT, COMPLEX
                default: '#FFFFFF',
                description: "The color when the photodiode check turns white"
            },
             black: {
                type: jsPsych.plugins.parameterType.STRING,
                default: '#000000',
                description: "The color when the photodiode check turns black"
            },
            wait_after: {
                type: jsPsych.plugins.parameterType.INT,
                default: 500,
                description: "waiting time before we close the trial"
            },
            background_color: {
                type: jsPsych.plugins.parameterType.STRING,
                default: '#848484',
                description: "default color of all circles"
            },
             target_pos_x: {
                type: jsPsych.plugins.parameterType.INT,
                default: [-5, -3, -1, 1, 3, 5],
                description: "The x position of the dots in the beginning"
            },
            target_pos_y: {
                type: jsPsych.plugins.parameterType.INT,
                default: [0, 0, 0, 0, 0, 0],
                description: "The y position of the dots in the beginning"
            },
            circle_radius: {
                type: jsPsych.plugins.parameterType.INT,
                default: 0.5,
                description: 'The size of each individual dot.',
            },
            pixel_per_degree: {
                type:jsPsych.plugins.parameterType.INT,
                default: 3.1415926535 * 10,
                description: 'How many pixels represent on degree visual angle'
            },
            accepted_touch_distance: {
                type:jsPsych.plugins.parameterType.INT,
                default: 2,
                description: 'How many dva around the target the touch response is accepted'
            },
        },
    };

    plugin.trial = function (display_element, trial) {

        // set the condition of the experiment
        let response_on = 'touchstart';

        // timing stuff
        let start_trial = false;

        // intialize parameters
        let photodiode_color = trial.background_color;
        let write_switch_time = false;
        let photodiode_switch_times = [];
        let photodiode_colors = [];
        let time_since_switch;

        let target_pos_x = trial.target_pos_x.map(elem => elem * trial.pixel_per_degree)
        let target_pos_y = trial.target_pos_y.map(elem => elem * trial.pixel_per_degree)

        let circle_radius = trial.circle_radius * trial.pixel_per_degree
        let accepted_touch_distance = trial.accepted_touch_distance * trial.pixel_per_degree

        // create a canvas   element
        display_element.innerHTML = '<canvas></canvas>';

        // initialize the canvas
        let canvas = document.querySelector('canvas');
        canvas.width = window.outerWidth;
        canvas.height = window.outerHeight;

        // get canvas context
        let c = canvas.getContext('2d');

        // initialize an animation request handler
        let AnimationRequest;

        // now, the interaction
        let screen_selected_at = {
            x: undefined,
            y: undefined,
        };

        let handle_interaction = function (event) {
            screen_selected_at.x = event.x || event.pageX || event.touches[0].clientX;
            screen_selected_at.y = event.y || event.pageY || event.touches[0].clientY;

            if (screen_selected_at.x < innerWidth / 2 + 30 && screen_selected_at.x > innerWidth / 2 - 30
                && screen_selected_at.y < innerHeight / 2 + 30 && screen_selected_at.y > innerHeight / 2 - 30
                && !start_trial) {
                start_trial = true
            }
        }

        let perform_updates = function (photodiode_square, photodiode_color) {
            photodiode_square.change_color(photodiode_color);

            for (let i = 0; i < target_circles.length; i++) {
                let x = target_circles[i].x_original;
                let y = target_circles[i].y_original;

                if (screen_selected_at.x - x < accepted_touch_distance && screen_selected_at.x - x > -accepted_touch_distance
                    && screen_selected_at.y - y < accepted_touch_distance && screen_selected_at.y - y > -accepted_touch_distance) {
                    console.log(i);
                    // remove circle
                    target_circles.splice(i, 1);
                    screen_selected_at.x = undefined;
                    screen_selected_at.y = undefined;
                }
            }
        };

        let get_current_position_and_color = function (timestamp) {
            write_switch_time = false;
            if (timestamp - time_since_switch > 500) {
                write_switch_time = true;
                photodiode_color = [trial.white, trial.black][+(photodiode_color === trial.white)];
                time_since_switch = timestamp;
            }
            return [photodiode_color];
        };

        let animate_setup = function (timestamp) {
            if (!start_trial) {
                requestAnimationFrame(animate_setup)
            } else {
                time_since_switch = timestamp;
                screen_selected_at.x = undefined;
                screen_selected_at.x = undefined;
                requestAnimationFrame(animate)
            }
            c.clearRect(0, 0, innerWidth, innerHeight);
            background_rect.draw();
            photodiode_check.draw();
            rectangle.draw();
        }

        let animate = function (timestamp) {
            AnimationRequest = requestAnimationFrame(animate);
            let repaint_values = get_current_position_and_color(timestamp);
            perform_updates(photodiode_check, repaint_values[0]);
            c.clearRect(0, 0, innerWidth, innerHeight);
            background_rect.draw();
            photodiode_check.draw();
            for (let i = 0; i < target_circles.length; i++) {
                target_circles[i].draw();
            }
            // check if trial is over
            if (target_circles.length === 0) {
                end_trial();
            }
            if (write_switch_time){
                photodiode_switch_times.push(timestamp);
                photodiode_colors.push(repaint_values[0]);
            }
        };


        let end_trial = function () {
            // break request animation frame loop
            cancelAnimationFrame(AnimationRequest);

            // kill any remaining setTimeout handlers
            jsPsych.pluginAPI.clearAllTimeouts();
            // gather the data to store for the trial
            let trial_data = {
                "switch_times": photodiode_switch_times, // when the photodiode switched
                "switch_colors": photodiode_colors, // which color the photodiode had
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

        let rectangle = new Rectangle(
            innerWidth / 2 - 10,
            innerHeight / 2 - 10,
            20,
            20,
            trial.white,
            c
        );

        let target_circles = init_circles(
            target_pos_x,
            target_pos_x,
            target_pos_y,
            target_pos_y,
            circle_radius,
            trial.black,
            c
        );

        // draw the rectangle for the photodiode
        let photodiode_check = new Rectangle(
            innerWidth - 40,
            0,
            40,
            20,
            trial.white,
            c
        );

        window.addEventListener(response_on, handle_interaction);

        // load new stylesheet and start animating
        let href = "custom_styles.css"
        let head = document.getElementsByTagName('head')[0];
        let trialStyleSheet = document.createElement('link');

        trialStyleSheet.type = "text/css";
        trialStyleSheet.rel = "stylesheet";
        trialStyleSheet.href = href;

        trialStyleSheet.onload = function () {
            console.log('style sheet was loaded');
            requestAnimationFrame(animate_setup)
        }
        trialStyleSheet.onerror = function () {
            console.log('The style sheet for the trial could not load!')
        };
        head.appendChild(trialStyleSheet);
    }
    return plugin;
})();
