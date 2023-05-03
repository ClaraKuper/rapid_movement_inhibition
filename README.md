# Rapid Movement Inhibition

In a series of 3 studies, we investigated the effects of unpredictable sudden-onset events on the frequency 
of eye and hand movement execution. The events were either task-relevant (changes of the movement target position), 
or task-irrelevant (bright flashes).

Experiments were run through web browsers on participant's smartphones. We used [jsPsych 6][jpsy_link] and [jatos][jatos_link] 
to power our studies.

The baseline task in the study consisted of tapping on a sequence of 6 dots - from left to right - with one's index finger.
Participants were given 1.5 seconds to tap on all dots in the array. 

- In the **remote background** study, the task-irrelevant change was a bright flash in the background of the movement targets.
- In the **remote target** study, the task-irrelevant change was a bright flash at the location of the movement target.
- In the **inlab all** study, bothe conditions were repeated while participants performed the same task in the lab, with the same technical setup (personal mobile phones)

This repository contains the code for the online experiments (html and javascript) along with analysis code (python) and result files.

[jpsy_link]: https://www.jspsych.org/6.3/
[jatos_link]: https://www.jatos.org/