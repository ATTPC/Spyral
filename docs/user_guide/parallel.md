# Spyral in Parallel

One of the other important performance features of Spyral is parallel processing of runs. In `start_pipeline` function you can specify a range of runs to process as well as the number of processors. Each processor is an independent Python interpreter process, running it's own instance of Spyral. Below we'll describe a little bit about how this works and what some of the strenghts and limitations are.

## How it Works

The parent process (the one caused by you running Spyral in the terminal) examines your config, and then does some work to prepare for parallelization. First, it pre-generates all of the shared resources. This includes creating the workspace, interpolation meshes, and checking for the existence of data. It then gathers all of the runs you requested be processed and tries to balance the load on each processor, such that no one processor is being overloaded with data (this would make that processor take forever compared to the others, basically defeating the purpose of parallelizing Spyral).

The balancing works as follows: The runs are sorted by the size of their raw trace files. Runs are then given to processors in a snake method:

```txt
runs = [9, 0, 2, 1, 3, 4] in order of size

processor 0: <- 9 <- 4 => [9,4]
                |    ^
                V    |
processor 1: <- 0 <- 3 => [0,3]
                |    ^
                V    |
processor 2: <- 2 <- 1 => [2,1]
```

This way works generally well at load balancing (assuming all runs are equal in data complexity for analysis).

Each processor is given a queue it uses to communicate to the parent process. Messages are sent to the parent when the analysis status changes (new phase, increment progress, etc). The awesome [tqdm](https://github.com/tqdm/tqdm) library is used to monitor the progress of each child processor.

Each processor (including the parent) writes its own log file, which can be found in the `log/` directory of the workspace. This allows for error reporting and finer grained messaging, so that way when something goes wrong you can find out why!

## Shared Memory

In some select cases it can be advantageous to share memory between the Spyral processes. The main example of this is the interpolation mesh created as an asset for the InterpSolverPhase. Typically the mesh is several GB in size, and as  such it is very expensive to allocate a mesh per process (particularly if you want to run 50 processes!). To address this, Spyral allows Phases to allocate shared memory using the `multiprocessing.shared_memory` library. Each Phase can override the `create_shared_data` function inherited from the base PhaseLike; if this function isn't overriden, the phase doesn't utilize shared memory. In general, shared memory should be used sparingly. Shared memory is somewhat difficult to guarantee safety on; shared memory as implemented should be strictly read-only within the context of the phases. 

## Optimizing Performance

To get the best performance, the number of runs to be analyzed should be evenly divisible by the number of processors. Otherwise, by necessity some of the processors will  generally *have* to run longer.  Also its best if runs are all of uniform size (but that's not usually within our control).

In general more processors is more better, but this isn't always the case. The first limit is the total number of cores in your machine. You need `n_processors`+1 physical cores, otherwise things are guaranteed to slow down. Another consideration is your CPU architecture. Intel i-series and Apple ARM use P-core and E-core packages. P-cores are performance cores which hit higher clock speeds and are optimized for expensive tasks. E-cores are efficiency cores, which run slower but are less power hungry, best suited to background tasks. Spyral is best suited to P-cores; as such you should use only P-cores as the number of processors available.

The final consideration is memory. This is mostly important for the solving phase. The interpolation mesh can be quite large (somtimes GBs) will briefly need to be loaded *twice* into memory (from file into the shared memory). This means you should have at least as much RAM as two times the mesh size (leave room to spare fo the rest of the memory we allocate!).

If you are using Spyral in a job environment (SLURM, etc), you can run with the Pipeline with the `no_display` argument in `start_pipeline` set to `True`. This will avoid the overhead of SLURM writting all the progress bar prints to a file.
