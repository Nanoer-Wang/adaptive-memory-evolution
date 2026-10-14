"""Registered attribution controls; calibration never uses task outcomes."""
import hashlib
import math
from src.memory.b1_policy import Policy, METHODS as ORIGINAL_METHODS

METHODS=(*ORIGINAL_METHODS,'static_state_adaptive','replacement_matched','static_uniform_state')


class AttributionPolicy(Policy):
    def __init__(self,method,capacity=8,seed=11,coefficients=None,admission=1.,decay=.001):
        if not 0<=admission<=1:raise ValueError('Admission probability outside [0,1]')
        base={'static_state_adaptive':'adaptive_evolution','replacement_matched':'fixed_decay','static_uniform_state':'fixed_decay'}.get(method,method)
        super().__init__(base,capacity,seed,coefficients,decay=decay)
        self.method=method;self.admission=admission;self.reject_gate=False

    def snapshot(self):
        # Item fields are scalars/strings: exactly asdict without recursive copying.
        return [dict(i.__dict__) for i in sorted(self.items.values(),key=lambda i:i.item_id)]

    def observe(self,event):
        result=super().observe(event)
        if self.method=='static_state_adaptive' and not result['duplicate']:
            # Identical learned insertion state, frozen for the entire residency.
            self.items[event['item_id']].I=self.alpha*self.held_w
        return result

    def score(self,item):
        if self.method=='replacement_matched':
            if self.reject_gate and item.item_id==self.pending:return -math.inf
            return -self.decay*(self.now-item.observed_at)
        return super().score(item)

    def evict(self):
        if self.method=='replacement_matched' and len(self.items)>self.capacity:
            token=f'b1-admission:{self.seed}:{self.calls["observe"]}'.encode()
            uniform=int.from_bytes(hashlib.sha256(token).digest()[:8],'big')/2**64
            self.reject_gate=uniform>=self.admission
        result=super().evict()
        # Serialize the forced gate explicitly, never non-finite numbers.
        if self.method=='replacement_matched':
            result['admission_probability']=self.admission
            result['gate_rejected']=self.reject_gate
            if self.reject_gate:result['scores'][result['candidate']]=None
        self.reject_gate=False
        return result


def replacement_summary(result):
    m=result['metrics'];lives=result['lifetimes']
    return dict(eviction_count=m['eviction_count'],
                mean_observed_lifetime=sum(x['lifetime'] for x in lives)/len(lives),
                time_weighted_resident_age=m['time_weighted_resident_age'])


def matching_loss(candidate,target):
    # Targets are calibration means only; no accuracy/stale/retention/lag inputs.
    return sum(((candidate[k]-target[k])/max(target[k],1.))**2
               for k in ('eviction_count','mean_observed_lifetime'))
