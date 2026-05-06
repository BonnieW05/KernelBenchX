import ast

class _KernelVisitor(ast.NodeVisitor):
    """AST visitor that checks Triton kernel validity and forbidden calls."""
    def __init__(self):
        self.has_jit = False
        self.tl_calls = set()
        self.torch_calls = set()
        self._in_kernel = False
        self._torch_aliases = {'torch', 'F'}
    
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == 'torch' and alias.asname:
                self._torch_aliases.add(alias.asname)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module and 'torch' in node.module:
            for alias in node.names:
                self._torch_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        # Detect @triton.jit decorated functions.
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Attribute) and dec.attr == 'jit') or \
               (isinstance(dec, ast.Name) and dec.id == 'jit'):
                self.has_jit = True
                self._in_kernel = True
                self.generic_visit(node)
                self._in_kernel = False
                return
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            prefix = node.func.value.id
            attr = node.func.attr
            if prefix in {'tl', 'triton'} and self._in_kernel:
                self.tl_calls.add(f"tl.{attr}")
            if prefix in self._torch_aliases and self._in_kernel:
                self.torch_calls.add(f"torch.{attr}")
        self.generic_visit(node)


def check_triton_validity(code: str, speedup: float = None) -> dict:
    """
    Check code validity via AST inspection.

    We parse the code into an AST, find functions decorated with @triton.jit, then:
    - record tl.* / triton.* primitive usage inside kernels
    - flag any torch.* calls inside kernels
    """
    flags = []
    if not code or not code.strip():
        return {"flags": ["empty_code"], "is_suspect": True}
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"flags": ["syntax_error"], "is_suspect": True}
    
    visitor = _KernelVisitor()
    visitor.visit(tree)
    
    # Findings
    if not visitor.has_jit:
        flags.append("no_triton_jit")
    if len(visitor.tl_calls) < 2:
        flags.append("few_tl_primitives")
    if visitor.torch_calls:
        flags.append(f"torch_in_kernel:{','.join(sorted(visitor.torch_calls))}")
    
    # Speedup anomaly flags
    if speedup is not None:
        if speedup > 1.05 and len(visitor.tl_calls) == 0:
            flags.append("speedup_without_triton_ops")
        if speedup > 10:
            flags.append("extreme_speedup")
    
    return {
        "flags": flags,
        "is_suspect": bool(flags),
        "tl_primitives": list(visitor.tl_calls),
    }
