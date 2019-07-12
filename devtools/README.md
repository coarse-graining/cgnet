Developer best practices
==

Thank you for contributing to `cgnet`! Here are our recommended practices for contributing, in no particular order of importance.

Developer and merging info
--
The developers (@coarse-graining/developers) are Brooke ([@brookehus](https://github.com/brookehus)), Nick ([@nec4](https://github.com/nec4)), and Dominik ([@Dom1L](https://github.com/Dom1L)). Only developers have merge permissions to master.

- PRs from non-developers require 2/3 approving developer reviews.
- Major PRs from developers require 2/2 approving reviews from the other developers.
- Minor PRs from developers require 1/2 approving reviews from the other developers.
- For now, if it's not obvious, we'll just talk within each PR about whether it's "major" or "minor". If this seems to be a hassle, we can add labels or come up with another method.


PR best practices
--
- Since the repo cannot be forked, always develop in your own branch.
- Default to PRing to other people's branches, even if it seems obnoxious. Never commit to someone else's branch unless you have cleared it with them first!
- *Never merge your own PR, especially to master*, but in general to any branch.
- Never commit any kinds of non-coding files without discussing first.
- Run `nosetests` before requesting final reviews, and run `nosetests` whenever you review.
- Use the labels and milestones that we've made to categorize your PRs (and issues).
- Be nice and constructive with your feedback!
- Always give a passing review before merging, even if it's just "LGTM"! 

Dependency best practices
--
- Discuss with the @coarse-graining/developers about how and whether to incorporate a new dependency.
- As of now, we are not adding plotting utilities to the main code; this may change in the future!

Coding best practices (just a small subset!)
--
- *Always* use python 3 and `pytorch >= 1.0`!
- Use `pep8` formatting! Packages like `autopep8` can help with this.
- Add yourself to the contributors list at the top of the file.
- Classes are `CamelCase`, and functions `use_underscores`.
- Intra-code dependencies should be one-directional: e.g., `cgnet.network` can import from `cgnet.feature`, but not the other way around.
- Use descriptive variable names, documentation, and comments! You will thank yourself later.
- Don't hide problems! Be transparent and add notes about anything that comes up but remains unaddressed. 

Testing best practices
--
- All tests go in `cgnet.tests` and are not imported in `cgnet.tests.__init__.py`. The exception for soft dependencies is that a `tests` folder should be in the relevant directory; see `cgnet.molecule` for an example.
- The function must start with `test_`. Use `#` for comments instead of `"""`. 
- The purpose of tests is so that future development doesn't break existing methods. Write tests so that if someone down the road breaks your method, your test will tell them!
- Each test should only test one aspect of the code, so if it breaks, you know what is implicated.
- It's okay, even recommended, to copy and paste code between tests!
- Tests shouldn't replicate the code of the main package, but should obtain values in other ways.

Example notebook best practices
--
- Example notebooks should be limited in scope and cover just one topic in a tutorial way. If the notebook becomes long-winded, break it into multiple notebooks.
- Never commit a notebook without the output cleared (if you do this by accident, [squash the commits](https://github.com/wprig/wprig/wiki/How-to-squash-commits)).
