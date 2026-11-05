import os, ast, json, sys

def get_top_package(module_name):
    return module_name.split('.')[0]

project_root = os.path.abspath(os.path.dirname(__file__))
imports_map = {}
for root, dirs, files in os.walk(project_root):
    if '.venv' in dirs:
        dirs.remove('.venv')  # skip virtual env
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    node = ast.parse(f.read(), filename=file_path)
            except Exception:
                continue
            for n in ast.walk(node):
                if isinstance(n, ast.Import):
                    for alias in n.names:
                        pkg = get_top_package(alias.name)
                        imports_map.setdefault(pkg, set()).add(file_path)
                elif isinstance(n, ast.ImportFrom):
                    if n.module:
                        pkg = get_top_package(n.module)
                        imports_map.setdefault(pkg, set()).add(file_path)
# Convert sets to sorted lists
result = {k: sorted(list(v)) for k, v in imports_map.items()}
print(json.dumps(result, indent=2))
