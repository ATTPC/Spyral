# Frequently Asked Questions

## Where can I learn about a specific function/class/code in Spyral?

See the [API docs](api/index.md) for API level documentation.

## How can I contriubte to Spyral?

See [For Developers](CONTRIBUTING.md)

## I just installed Spyral and nothing is working

See the [User Guide](user_guide/getting_started.md)

## What do all these configuration parameters mean?

See [Configuration](user_guide/config/about.md)

## Which solver should I use?

See [Solving](user_guide/phases/solve.md)

## I have some old data I want to analyze, can I use Spyral?

If your data was taken in the era where the AT-TPC *did not* have the FRIBDAQ auxiliary
system, default Spyral will not be able to analyze your data. However, there are some
projects which have worked to make Spyral compatible with these datasets; see
[this repo from Zach Serikow](https://github.com/sigmanotation/e20009_analysis).

## I just ran Spyral and it seems like everything went well, but there is no output data/the results didn't change

Check the log files in your workspace (in the `<your_workspace>/log` directory). Spyral 
will try to catch-and-continue most errors that aren't directly related to reading the 
configuration and spawning the processes. This helps us not crash out if one single 
file out 100 files to be analyzed is not formated correctly, but can sometimes make it 
look like Spyral ran but didn't do anything. The logs should contain lines indicating 
if a crash happened and what was detected as a possible error.

## I am using Linux and Spyral crashes when I try to use more than one or two processes

This is most likely related to 
[this issue](https://github.com/ATTPC/Spyral/issues/135). You can modify your script 
with the following edit

```python

import os # Add this import
...

if name == 'main':
    os.system('taskset -cp 0-%d %s' % (n_processes, os.getpid())) # Add this line
    multiprocessing.set_start_method("spawn")
    main()
```

This is due to a known issue with multiprocessing and some versions of OpenBLAS, where 
the processor affinity is overriden by OpenBLAS.

## My experiment did not measure the window time, how do I calibrate my point clouds?

Spyral provides a function [`calculate_window_time`](api/core/config.md)
which can be used to calculate the window time bucket from a known drift velocity, 
GETDAQ sampling frequency, and micromegas time bucket. To import this function use:

```python
from attpc_spyral import calculate_window_time
```
