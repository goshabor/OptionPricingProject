from scipy.optimize import differential_evolution, minimize

def progress(xk, convergence):
    print(f"[DE] conv={convergence:.3f} best_x≈{xk}")
    return False

def local_progress(xk, state=None):
    print(f"[LOCAL] x≈{xk}")

def run_differential_evolution_with_polish(class_, args, bounds, constraint=None):
    kwargs = dict(
        func=class_.objective_func,
        args=() if args is None else args,
        strategy='best2exp',
        bounds=bounds,
        tol=1e-4,
        atol=1e-4,
        maxiter=1000,
        mutation=(0.5, 0.7),
        recombination=0.7,
        updating='immediate',
        workers=1,
        polish=False,        
        disp=True,
        callback=progress,
    )

    kwargs['constraints'] = constraint
    result_de = differential_evolution(**kwargs)

    print("\n=== Finished DE stage ===")
    print(f"DE best fun = {result_de.fun}")
    print(f"DE best x = {result_de.x}")
    return result_de