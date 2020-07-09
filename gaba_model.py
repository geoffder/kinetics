import numpy as np
import matplotlib.pyplot as plt


class GabaKineticModel:
    def __init__(self, pulses, dt=.01, tstop=100):
        self.dt = dt
        self.tstop = tstop
        self.time = np.arange(tstop // dt) * dt
        self.set_agonist(pulses)
        self.reset_state()
        self.set_rates()
        
    def set_agonist(self, pulses):
        """Takes list of triples (onset, duration, concentration) of agonist
        pulses. Build ndarray representing agonist concentration over time."""
        self.agonist = np.zeros_like(self.time)
        for (on, dur, conc) in pulses:
            self.agonist[(self.time >= on) * (self.time < (on + dur))] += conc

    def reset_state(self):
        """All receptors start in the unbound ready state."""
        self.state = {
            "R": 1.,
            "AR": 0.,
            "AR*": 0.,
            "AD": 0.,
            "A2R": 0.,
            "A2R*": 0.,
            "A2D": 0.
        }

    def set_rates(self, updates={}):
        """State transition rates.
        - k_on and k_off are agonist sensitive, governing transition between
        ready states (R <-> AR <-> A2R).
        - (b)eta and (a)lpha rates govern transition to and from open states
        (AR <-> AR* and A2R <-> A2R*).
        - (d)eactivation and (r)eactivation rates govern transition to and from
        deactivated state (AR <-> AD and A2R <-> A2D)"""
        self.rates = {
            "k_on": 8.,
            "k_off": 0.12,
            "b1": 0.04,
            "a1": 0.2,
            "d1": 0.013,
            "r1": 0.0013,
            "b2": 3.45,
            "a2": 1.0,
            "d2": 1.45,
            "r2": 0.1,
        }
        for k, v in updates:
            self.rates[k] = v

    def step(self, ago):
        # calculate deltas along each of the state transitions
        k1_on = self.rates["k_on"] * 2 * self.dt * ago * self.state["R"]
        k1_off = self.rates["k_off"] * self.dt * self.state["AR"]
        b1 = self.rates["b1"] * self.dt * self.state["AR"]
        a1 = self.rates["a1"] * self.dt * self.state["AR*"]
        d1 = self.rates["d1"] * self.dt * self.state["AR"]
        r1 = self.rates["r1"] * self.dt * self.state["AD"]
        k2_on = self.rates["k_on"] * self.dt * ago * self.state["AR"]
        k2_off = self.rates["k_off"] * 2 * self.dt * self.state["A2R"]
        b2 = self.rates["b2"] * self.dt * self.state["A2R"]
        a2 = self.rates["a2"] * self.dt * self.state["A2R*"]
        d2 = self.rates["d2"] * self.dt * self.state["A2R"]
        r2 = self.rates["r2"] * self.dt * self.state["A2D"]

        # update proportion of population in each state
        self.state["R"] += k1_off - k1_on
        self.state["AR"] += k1_on + a1 + r1 + k2_off - k1_off - k2_on - b1 - d1
        self.state["AR*"] += b1 - a1
        self.state["AD"] += d1 - r1
        self.state["A2R"] += k2_on + a2 + r2 - k2_off - b2 - d2
        self.state["A2R*"] += b2 - a2
        self.state["A2D"] += d2 - r2

        self.state = {k: max(0, min(1, v)) for k, v in self.state.items()}

    def run(self):
        recs = {k: [] for k in self.state.keys()}
        for concentration in self.agonist:
            self.step(concentration)
            for k, v in self.state.items():
                recs[k].append(v)
        return {k: np.array(v) for k, v in recs.items()}

if __name__ == "__main__":
    base_pth = "/mnt/Data/NEURONoutput"

    model = GabaKineticModel([(0, 2, 10)])
    recs = model.run()
    open_pop = recs["AR*"] + recs["A2R*"]

    plt.plot(model.time, open_pop)
    plt.show()
