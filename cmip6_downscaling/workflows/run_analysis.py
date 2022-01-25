import papermill as pm

out_path = fs.

pm.execute_notebook(
   '../../',
   'path/to/output.ipynb',
   parameters=dict(alpha=0.6, ratio=0.1)
)
