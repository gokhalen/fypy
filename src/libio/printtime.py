def printtime(tpre=None,tassem=None,tsolve=None,tout=None,ttotal=None,treduc=None,tstrain=None,tstrainsolve=None,treducstrain=None,digits=3):
    print('-'*80)
    print(f'Preprocessing time \t\t= {tpre:0.{digits}f}s \t {(tpre/ttotal)*100:>10.{digits}f}%')    
    print(f'Assembly time \t\t\t= {tassem:0.{digits}f}s  \t {(tassem/ttotal)*100:>10.{digits}f}%')
    print(f'Displacement Solver time \t= {tsolve:0.{digits}f}s \t {(tsolve/ttotal)*100:>10.{digits}f}%')
    print(f'Strain Assembly time \t\t= {tstrain:0.{digits}f}s  \t {(tstrain/ttotal)*100:>10.{digits}f}%')
    print(f'Strain Solve time \t\t= {tstrainsolve:0.{digits}f}s  \t {(tstrainsolve/ttotal)*100:>10.{digits}f}%')
    print(f'Output time \t\t\t= {tout:0.{digits}f}s \t {(tout/ttotal)*100:>10.{digits}f}%')
    print(f'Total time \t\t\t= {ttotal:0.{digits}f}s\t {(ttotal/ttotal)*100:>10.{digits}f}%')
    print('-'*80)
    print(f'Reduction time for displacement assembly (approx for multiproc) = {treduc:0.{digits}f}s,\n\
    {(treduc/ttotal)*100:0.{digits}f}% (of Total time) ')
    print(f'Reduction time for strain assembly (approx for multiproc) = {treducstrain:0.{digits}f}s,\n\
    {(treducstrain/ttotal)*100:0.{digits}f}% (of Total time) ')
    print('-'*80)
        
    
