# NeuroTech-ML

## Notes
- "data" directory will hold all our data. We will keep data files labelled with the date to make it easy to follow.
- Currently, use the "001_trial1_right_log_18-09-46-931825.txt" file. It's raw csv data with 3 columns: timestamp(ms) | power | keypress.
Note that every keypress comes up as a duplicate, despite one press. Just take the first one. With this limitation in mind, just use this data to produce preliminary visualizations for transforms. We will get more data as we go!

## Siggy workflow reminders
Within the "siggy" directory, the next sublevel will be the "main" directory and individual directories. Submit your code to your individual directory. Please use branches and pull requests. Feel free to merge your pull request for your individual directories, but for "main" just leave the pull request for approval. Ask Miasya if you have any questions/suggestions about this!

## Siggy upcoming goals:
  * Upload invidual work on generating spectral features from EMG. Miasya will combine everything so it follows a standard format (i.e. ensure ease of use)
  * Move on from signal classification for now. Jump to hidden markov chains. Target: demo progress Sunday the 16th 

Task description (markov)
  * Read up!
  * Use corpus of most commonly used English words
  * Change words (strings) to finger mapping
