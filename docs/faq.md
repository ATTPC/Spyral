# Frequently Asked Questions

## Where can I learn about a specific function/class/code in Spyral?

See the [API docs](api/index.md) for API level documentation.

## How can I contriubte to Spyral?

See [For Developers](for_devs.md)

## I just installed Spyral and nothing is working

See the [User Guide](user_guide/getting_started.md)

## What do all these configuration parameters mean?

See [Configuration](user_guide/config/about.md)

## I have some old data I want to analyze, can I use Spyral?

In general, yes. Spyral has a legacy mode that supports analyzing data taken before AT-TPC's data acquisition was split (pre-FRIBDAQ): see [here](user_guide/phases/point_cloud.md) for more details. However, at this time, it only supports analyzing the ion chamber data out of CoBo 10. Any other auxilary signals will have to be extended or requested.

## I just ran Spyral and it seems like everything went well, but there is no output data/the results didn't change

Check the log files in your workspace (in the `<your_workspace>/log` directory). Spyral will try to catch-and-continue most errors that aren't directly related to reading the configuration and spawning the processes. This helps us not crash out if one single file out 100 files to be analyzed is not formated correctly, but can sometimes make it look like Spyral ran but didn't do anything. The logs should contain lines indicating if a crash happened and what was detected as a possible error.

## I am using Linux and Spyral crashes when I try to use more than one or two processes

This is most likely related to [this issue](https://github.com/ATTPC/Spyral/issues/135). You can modify your script with the following edit

```python

import os # Add this import
...

if name == 'main':
    os.system('taskset -cp 0-%d %s' % (n_processes, os.getpid())) # Add this line
    multiprocessing.set_start_method("spawn")
    main()
```

This is due to a known issue with multiprocessing and some versions of OpenBLAS, where the processor affinity is overriden by OpenBLAS.
