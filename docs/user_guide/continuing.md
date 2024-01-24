# After Spyral

Once you're done with all the Spyral phases, the true experiment analysis begins. Spyral doesn't provide anything further, as the analysis after fitting becomes very experiment specific. But we can layout the general next steps.

## Where is my data?

The solver phase outputs data to the `physics/` directory of the workspace. Data files are named by run number and particle symbol (from the used particle ID gate). These are parquet files, which is a generic dataframe format. You can use any of the dataframe libraries to load this; pick your  favorite! We love [polars](https://pola.rs/), but [pandas](https://pandas.pydata.org/) is of course the industry standard.

If you need to access the point clouds or clusters, see the relevant directories and code, which describe those data formats.

## Can I use any of the tools from Spyral

The [spyral-utils](https://github.com/gwm17/spyral-utils/) library is a subset of the original Spyral code that we extracted and packaged because we found it really useful for analyzing all kinds of data not just Spyral. This contains things like nuclear masses, energy loss through the [pycatima](https://pypi.org/project/pycatima/) library for both gas and solid materials, very basic four-vector analysis, and some histogramming utilities. Spyral also does have a [notebook](notebooks.md) that shows how we use some of these tools.

## Final Thoughts

Since Spyral is a framework, it tries to be as generic as possible. This does mean that for some data, heavy customization will be needed to get the best possible results. But that's why we wrote it in Python and made these documents! Hopefully that makes it easier for people to understand how Spyral works and make it work for their needs.
