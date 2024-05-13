# For Developers

Spyral is always looking for input and improvements! If you would like to join in our collaboration, please notify a maintainer of the [AT-TPC GitHub](https://github.com/attpc/) or Spyral. They can add you as a collaborator and setup a development branch for you in Spyral. New ideas are always welcome! Below are some outlines for working in the Spyral collaboration.

## Required Development Tools

### PDM

Spyral uses [PDM](https://pdm-project.org/en/latest/) as it's project management tool.

## Pull Requests

### Statement of Intent

Please state in the pull request what the goal of is of this modification to Spyral and what was changed to accomplish this. If there is no statement of intent, the pull request will not be accepted.

### Statement of Dependencies

Please state in the pull request what new dependencies (if any) were added and please make sure that these new dependencies were added to the Spyral `requirements.txt` with a version pinned. If the dependencies were not stated, the pull request will not be accepted. Please check to make sure that any new dependencies are compliant with the [GPL3](https://www.gnu.org/licenses/gpl-3.0.en.html) license which is used by Spyral.

### Review

Any pull request will require at least one reviewer.

## Code Requirements

Below are some of the requirements for any code added to Spyral:

### Docstrings

Please provide docstrings for each function and class. Dataclasses do not necessarily need docstrings. This helps other developers understand what the purpose and usage code. Check out some of the docstrings in Spyral to get a feel for the expected format.

### Type Hints

Please use [type hints](https://docs.python.org/3/library/typing.html) to annotate your code where applicable. This includes function arguments, return values, and any variables where the type might be ambiguous. In general, a good rule to follow is: can your IDE determine the type of the variable? If it can it doesn't need a type hint. If it can't, the variable needs a type hint. The Any type hint is allowed in some specific cases. Functions which return None do not need a return type hint.

In some places in the code you may notice the comment `# type: ignore`. The Spyral development team uses type checking to help detect and eliminate issues in the codebase before they get deployed. However, many libraries don't provide a level of typing which allows for type checking. This comment will disable type checking and static type analysis for that line. It should only be used when it has been confirmed through testing that that line behaves as expected.

### Formatting

Spyral uses the black formatter. The appropriate version of black is included in the requirements.txt file, so simply install and everything should be good to go.

### Files

Please try to keep files from being monster 1000 lines of code documents. This is not a hard and fast rule, but in general files should contain a unqiue individual unit of code which interfaces with the rest of the framework. There can be execptions.

### Unit Tests

Where reasonable, please provide unit tests in the tests directory for code you add. Obviously, as a data analysis library, there are limitations to what tests we can automate. But they are really helpful for making sure things don't break.

### Documentation

If you decide to contribute a new major feature to Spyral, please prepeare some documentation to be added to this site. It should at least outline what new configuration parameters are exposed and what effect this may have upon the data. Documentation should be contributed in the form of Markdown files in the `docs` directory. Our documenation is built using the amazing [MkDocs](https://www.mkdocs.org/) and the [MkDocs-Material](https://squidfunk.github.io/mkdocs-material/) theme.

## Final Thoughts

Below is an example pull request descripton:

```txt
Intent:
Fix a bug in configuration parsing in Spyral and switch to YAML

Dependencies Added:
pyyaml
```

Feel free to use this as a simple template if you wish!

If you find that you need to really customize your Spyral to fit your specific experiment, please consider forking the parent Spyral repository. This will allow you to still get all the power of GitHub and version control, without having to make your code necessarily complaint with Spyral's experiment agnostic goals.
