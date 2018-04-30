import os
import pstats
from pyprof2calltree import visualize, convert

# run in the console

base_path = os.path.dirname(__file__)
prof_res_path = os.path.join(base_path, 'results', 'prof.stats')
stats = pstats.Stats(prof_res_path)
stats.print_stats()

# run kcachegrind and save the file
visualize(prof_res_path)
convert(prof_res_path, os.path.join(prof_res_path, f'prof.kgrind'))