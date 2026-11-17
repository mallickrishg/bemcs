"""Read percent-format ("# %%") example scripts as notebooks for MyST-NB.

jupytext gives notebooks without kernel metadata; add it so code cells are highlighted as
Python and executed with the python3 kernel.
"""

import jupytext


def reads(text, **kwargs):
    nb = jupytext.reads(text, fmt={"extension": ".py", "format_name": "percent", "comment_magics": False})
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb
