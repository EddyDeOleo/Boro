import os, ast, json
imports = {}
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    tree = ast.parse(fh.read(), filename=path)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        pkg = n.name.split('.')[0]
                        imports.setdefault(pkg, set()).add(path)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        pkg = node.module.split('.')[0]
                        imports.setdefault(pkg, set()).add(path)
json_output = {k: sorted(list(v)) for k, v in imports.items()}
print(json.dumps(json_output, indent=2))
