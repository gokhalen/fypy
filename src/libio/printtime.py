def printtime(tpre=None,tassem=None,tsolve=None,tout=None,ttotal=None,treduc=None,digits=3):
    print('-'*80)
    print(f'Preprocessing time \t= {tpre:0.{digits}f}s \t {(tpre/ttotal)*100:>10.{digits}f}%')    
    print(f'Assembly time \t\t= {tassem:0.{digits}f}s  \t {(tassem/ttotal)*100:>10.{digits}f}%')
    print(f'Solver time \t\t= {tsolve:0.{digits}f}s \t {(tsolve/ttotal)*100:>10.{digits}f}%')
    print(f'Output time \t\t= {tout:0.{digits}f}s \t {(tout/ttotal)*100:>10.{digits}f}%')
    print(f'Total time \t\t= {ttotal:0.{digits}f}s\t {(ttotal/ttotal)*100:>10.{digits}f}%')
    print('-'*80)
    print(f'Reduction time (approx for multiproc) = {treduc:0.{digits}f}s,\
    {(treduc/ttotal)*100:0.{digits}f}% (of Total time) ')
    print('-'*80)
        
    
