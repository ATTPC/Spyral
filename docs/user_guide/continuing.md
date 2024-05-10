# After Spyral

Once you're done with all the Spyral phases, the true experiment analysis begins. Spyral doesn't provide anything further, as the analysis after fitting becomes very experiment specific.

## Where is my data?

All of the data created by Spyral is stored at your workspace path. Each phase has a folder with it's name. Data files are named by run number (and sometimes an isotope symbol). Default spyral produces HDF5 files and parquet files, which is a generic dataframe format. You can use any of the dataframe libraries to load parquet files; pick your  favorite! We love [polars](https://pola.rs/), but [pandas](https://pandas.pydata.org/) is of course the industry standard.

## Can I use any of the tools from Spyral

The [spyral-utils](https://github.com/gwm17/spyral-utils/) library is a subset of the original Spyral code that we extracted and packaged because we found it really useful for analyzing all kinds of data not just Spyral. This contains things like nuclear masses, energy loss through the [pycatima](https://pypi.org/project/pycatima/) library for both gas and solid materials, very basic four-vector analysis, and some histogramming utilities. You can of course always use Spyral itself, but it can be a very heavy library.
