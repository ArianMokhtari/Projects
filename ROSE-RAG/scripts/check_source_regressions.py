"""Check the corrected source edge cases without launching experiment drivers.

Requires the numerical source dependencies; does not change input/reference files.
"""
from pathlib import Path
import __future__
import ast
import hashlib
import json
import sys
import types
from types import SimpleNamespace
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
CHECKS=[]


def require(name,passed):
    CHECKS.append({'check':name,'passed':bool(passed)})
    if not passed:
        raise AssertionError(name)


def load(fragment):
    path=next((ROOT/'source_archive').glob(fragment))
    nb=json.loads(path.read_text())
    source='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    name='_rose_regression_'+hashlib.sha256(path.name.encode()).hexdigest()[:10]
    mod=types.ModuleType(name);sys.modules[name]=mod
    safe=(ast.Constant,ast.List,ast.Tuple,ast.Dict,ast.Set,ast.BinOp,ast.UnaryOp,
          ast.Name,ast.Load,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub,ast.UAdd)
    for node in ast.parse(source).body:
        run=isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))
        if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('IPython'):
            run=False
        if isinstance(node,ast.Assign):
            run=all(isinstance(n,safe) for n in ast.walk(node.value))
        if run:
            try:
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec',
                             flags=__future__.annotations.compiler_flag),mod.__dict__)
            except NameError:
                if not isinstance(node,ast.Assign):raise
    return mod


for fragment in ['*Experiment1_vector*.ipynb','*Experiment1B*.ipynb','*Experiment2*.ipynb',
                 '*Experiment3_Misspecification*.ipynb','*Experiment3B*.ipynb']:
    m=load(fragment)
    # Unnormalized beliefs represent the same conditional problem at every scale.
    # The exact one-query value is .01 + (1/30 + .3 * 23/30).
    nominal_channels=np.array([[[.1,.9],[.3,.7]]])
    for scale in [1.,1e-12,1e-16,1e-20,1e-100]:
        value=m.nominal_dp_from_state(np.array([1.,2.])*scale,nominal_channels,
                                      np.array([.01]),.3,(0,))
        require(fragment+' nominal belief scale '+str(scale),abs(value-(.01+1/30+.3*23/30))<1e-12)
    zero=m.make_channel(0.,0.)[None,None,:,:]
    for label,channels,projection in [
        ('zero',zero,[[(0.,0.)]]),
        ('tiny positive',m.make_channel(1e-20,2e-20)[None,None,:,:],[[(1e-20,2e-20)]])]:
        hcr,_=m.hcr_dp(.5,channels,np.array([.01]),.3)
        rect,_=m.rectangular_dp(.5,projection,np.array([.01]),.3)
        require(fragment+' '+label+' HCR',abs(hcr[0]-.3)<1e-12)
        require(fragment+' '+label+' Rectangular',abs(rect[0]-.3)<1e-12)
    if hasattr(m,'wilson_interval'):
        for confidence in [.90,.95,.99]:
            lo,hi=m.wilson_interval(50,100,confidence)
            z=norm.ppf((1+confidence)/2);half=z*np.sqrt(.25/100+z*z/(4*100**2))/(1+z*z/100)
            require('Wilson confidence '+str(confidence),max(abs(lo-(.5-half)),abs(hi-(.5+half)))<1e-12)
    if hasattr(m,'solve_restricted_policy_mixture'):
        original=m.linprog
        try:
            # A feasible LP result with tiny negative roundoff and a tiny positive mass.
            m.linprog=lambda *a,**k:SimpleNamespace(success=True,x=np.array([-1e-11,1e-13,1+1e-11-1e-13,.1]))
            weights,value=m.solve_restricted_policy_mixture(np.full((3,2),.1))
            require(fragment+' nonnegative normalized mixture',(weights>=0).all() and abs(weights.sum()-1)<1e-12)
            require(fragment+' retains positive small mass',weights[1]>0)
            m.linprog=lambda *a,**k:SimpleNamespace(success=True,x=np.array([-.01,.01,1.,.1]))
            try:
                m.solve_restricted_policy_mixture(np.full((3,2),.1))
            except RuntimeError:rejected=True
            else:rejected=False
            require(fragment+' rejects materially negative mixture',rejected)
        finally:m.linprog=original

g=load('*Issue4_Scalability*.ipynb')
# Distinct rare alphabet observations have different conditional terminal risks.
# Rounding their unnormalized likelihoods to 14 places used to merge all three.
rare=np.array([[[1e-16,9e-16,1.-1e-15],[6e-16,14e-16,1.-2e-15]]])
_,rare_memo=g.nominal_dp_general(.5,rare,np.array([.01]),.3)
for evidence in range(3):
    omega=np.array([.5,.5])*rare[0,:,evidence]
    key=(tuple(float(v) for v in omega),())
    posterior=float(omega[1]/omega.sum())
    expected=min(posterior,1-posterior,.3)
    require('general nominal rare state '+str(evidence),key in rare_memo and abs(rare_memo[key][0]-expected)<1e-12)
channels=np.array([[[.1,.9,0.],[.9,.1,0.]]])
costs=np.array([.01])
nominal,memo=g.nominal_dp_general(.5,channels,costs,.3)
tree=g.nominal_tree_from_memo(.5,channels,memo)
require('general nominal zero-mass observation reconstruction',tree[0]=='query_0' and tree[1][2]==('abstain',None))
for name,call in [('HCR',g.hcr_dp_general),('Rectangular',g.rectangular_dp_general)]:
    result=call(.5,channels[None,:,:,:],costs,.3)[0]
    require('general '+name+' structural-zero alphabet',abs(result[0]-.11)<1e-12)

original=g.linprog
try:
    g.linprog=lambda *a,**k:SimpleNamespace(success=True,x=np.array([-1e-11,1e-13,1+1e-11-1e-13,.1]))
    weights,value=g.solve_restricted_policy_mixture_general(np.full((3,2),.1))
    require('general nonnegative normalized mixture',(weights>=0).all() and abs(weights.sum()-1)<1e-12)
    require('general retains positive small mass',weights[1]>0)
    g.linprog=lambda *a,**k:SimpleNamespace(success=True,x=np.array([-.01,.01,1.,.1]))
    try:g.solve_restricted_policy_mixture_general(np.full((3,2),.1))
    except RuntimeError:rejected=True
    else:rejected=False
    require('general rejects materially negative mixture',rejected)
finally:g.linprog=original

print(json.dumps({'passed':True,'checks':len(CHECKS),'results':CHECKS},indent=2))
