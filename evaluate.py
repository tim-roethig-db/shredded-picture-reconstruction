import blackbox

oracle = blackbox.BlackBox(shredded_path='shredded.png', orig_path='original.png')

permutation = list(range(128))

print(oracle.evaluate_solution(permutation))

oracle.show_solution(permutation)