import stim,sinter
import matplotlib.pyplot as plt
from code_construction.code_construction import CodeConstructor,CSSCode
import numpy as np
from ldpc.sinter_decoders.sinter_lsd_decoder import SinterLsdDecoder

class CircuitGenerator:
    """
    Generic CSS-code Stim circuit generator (robust layering).
    - One circuit = one basis memory experiment (measure_data in {'Z','X'}).
    - Each round has TWO sequential blocks according to `order`:
        Block(which):
          reset anc(which)  -> couplings(which) -> measure(which)
          immediately add time-diff DETECTORs for *this* block versus previous round.
    - Burn-in rounds: do the same but don't add DETECTORs.
    - Data is prepared ONCE at the beginning; never reset mid-circuit.
    - Optional end-caps only for the same basis as final readout.
    - Noise:
        pd: per-round DEPOLARIZE1 on data (applied at round start)
        pg: DEPOLARIZE2 after each CNOT
        ps: pre-measure flip (Z->X_ERROR, X->Z_ERROR)
        pp: post-prep flip on prepared qubits (Z->Z_ERROR after R, X->X_ERROR after RX)
    """

    def __init__(self, css):
        self.hx = np.array(css.hx, dtype=np.uint8) % 2
        self.hz = np.array(css.hz, dtype=np.uint8) % 2
        self.n  = int(self.hx.shape[1])
        assert self.hz.shape[1] == self.n
        self.mX = int(self.hx.shape[0])
        self.mZ = int(self.hz.shape[0])

        self.Lx = None if getattr(css, 'lx', None) is None else (np.array(css.lx, dtype=np.uint8) % 2)
        self.Lz = None if getattr(css, 'lz', None) is None else (np.array(css.lz, dtype=np.uint8) % 2)
        if self.Lx is not None: assert self.Lx.shape[1] == self.n
        if self.Lz is not None: assert self.Lz.shape[1] == self.n

        self.data_start = 0
        self.data_end   = self.n
        self.ancZ_start = self.data_end
        self.ancZ_end   = self.ancZ_start + self.mZ
        self.ancX_start = self.ancZ_end
        self.ancX_end   = self.ancX_start + self.mX


    def _supp_hz(self, j): return np.flatnonzero(self.hz[j]).tolist()
    def _supp_hx(self, j): return np.flatnonzero(self.hx[j]).tolist()
    def _supp_Lz(self, j): return np.flatnonzero(self.Lz[j]).tolist()
    def _supp_Lx(self, j): return np.flatnonzero(self.Lx[j]).tolist()

    def dataqubit_preparation(self,pp,pd,data_prep,measure_data):
        c = stim.Circuit()
        data_idxs = list(range(self.data_start, self.data_end))
        if self.n > 0:
            prep_mode = data_prep if data_prep != 'auto' else ('X' if measure_data == 'X' else 'Z')
            if prep_mode == 'Z':
                c.append_operation('R', data_idxs)
                c.append_operation('X_ERROR', data_idxs, pp)
            elif prep_mode == 'X':
                c.append_operation('RX', data_idxs)
                c.append_operation('Z_ERROR', data_idxs, pp)
            elif prep_mode != 'none':
                raise ValueError(f"invalid data_prep={data_prep!r}")
            if pd > 0:
                c.append_operation('DEPOLARIZE1', data_idxs, pd)
        return c
    def measureancilla_preparation(self,pp):
        c = stim.Circuit()
        ancX = list(range(self.ancX_start,self.ancX_end))
        ancZ = list(range(self.ancZ_start,self.ancZ_end))
        c.append_operation('R',ancX)
        c.append_operation('R',ancZ)
        c.append_operation('X_ERROR',ancX,pp)
        c.append_operation('X_ERROR',ancZ,pp)
        return c
    def cnotcircuit(self,pd,pg):
        ancX = list(range(self.ancX_start,self.ancX_end))
        ancZ = list(range(self.ancZ_start,self.ancZ_end))
        data_idxs = list(range(self.data_start, self.data_end))

        c = stim.Circuit()
        c.append_operation('H',ancX)
        c.append_operation('DEPOLARIZE1',ancX,pd)
        c.append_operation('TICK')
        cnot_gate_set = []
        ordered_cnot_gate_set = []
        # get all the cnot gates
        for tar in range(self.mZ):
            ctrls = self._supp_hz(tar)
            for ctrl in ctrls:
                cnot_gate_set.append((data_idxs[ctrl],ancZ[tar]))
 

        # arranging cnot gates
        for cnot in cnot_gate_set:
            placed_flag = False
            
            for order in range(len(ordered_cnot_gate_set)):
                placable_flag = True
                for arranged_cnot in ordered_cnot_gate_set[order]:
                    if arranged_cnot[0] == cnot[0] or arranged_cnot[0] == cnot[1] or arranged_cnot[1] == cnot[0] or arranged_cnot[1] == cnot[1]:
                        placable_flag = False
                if placable_flag:
                    ordered_cnot_gate_set[order].append(cnot)
                    placed_flag = True
                    break
            if not placed_flag:
                ordered_cnot_gate_set.append([cnot])

        # adding cnots to the circuit
        for cnots in ordered_cnot_gate_set:
            for cnot in cnots:
                c.append_operation('CX',cnot)
                c.append_operation('DEPOLARIZE2',cnot)
            c.append_operation('TICK')
        
        cnot_gate_set= []
        ordered_cnot_gate_set = []
        for ctrl in range(self.mX):
            tars = self._supp_hx(ctrl)
            for tar in tars:
                cnot_gate_set.append((ancX[ctrl],data_idxs[tar]))
        for cnot in cnot_gate_set:
            placed_flag = False
            
            for order in range(len(ordered_cnot_gate_set)):
                placable_flag = True
                for arranged_cnot in ordered_cnot_gate_set[order]:
                    if arranged_cnot[0] == cnot[0] or arranged_cnot[0] == cnot[1] or arranged_cnot[1] == cnot[0] or arranged_cnot[1] == cnot[1]:
                        placable_flag = False
                if placable_flag:
                    ordered_cnot_gate_set[order].append(cnot)
                    placed_flag = True
                    break
            if not placed_flag:
                ordered_cnot_gate_set.append([cnot])

        # adding cnots to the circuit
        for cnots in ordered_cnot_gate_set:
            for cnot in cnots:
                c.append_operation('CX',cnot)
                c.append_operation('DEPOLARIZE2',cnot)
            c.append_operation('TICK')

        c.append_operation('H',ancX)
        c.append_operation('DEPOLARIZE1',ancX,pd)
        c.append_operation('TICK')

        return c
    def measurements(self,ps):
        ancX = list(range(self.ancX_start,self.ancX_end))
        ancZ = list(range(self.ancZ_start,self.ancZ_end))
        c = stim.Circuit()
        c.append_operation('X_ERROR',ancZ,ps)
        c.append_operation('X_ERROR',ancX,ps)
        c.append_operation('MR',ancZ)
        c.append_operation('MR',ancX)
        c.append_operation('X_ERROR',ancZ,ps)
        c.append_operation('X_ERROR',ancX,ps)
        return c


    def generate(
        self,
        rounds=1, 
        noise_model=None,
        burn_in_rounds=1,
        measure_data='Z',         # 'Z' or 'X'
        add_observable=True,
        data_prep='auto',         # 'auto'|'Z'|'X'|'none'
        add_endcaps=False         # start False; turn True 
    ) -> stim.Circuit:

        if noise_model is None:
            noise_model = {'pd':0.0, 'pg':0.0, 'ps':0.0, 'pp':0.0}
        pd = float(noise_model.get('pd', 0.0))
        pg = float(noise_model.get('pg', 0.0))
        ps = float(noise_model.get('ps', 0.0))
        pp = float(noise_model.get('pp', 0.0))

        assert measure_data in ('Z', 'X')
        assert rounds >= 1 and burn_in_rounds >= 0

        c = stim.Circuit()
        data_idxs = list(range(self.data_start, self.data_end))

        # ---- data preparation (once) ----
        c += self.dataqubit_preparation(pp,pd,data_prep,measure_data)
        c += self.measureancilla_preparation(pp)
        c.append_operation('DEPOLARIZE1',data_idxs,pd)
        c += self.cnotcircuit(pd,pg)
        c += self.measurements(ps)
        if measure_data == 'Z':
            for i in range(self.mZ):
                c.append_operation('DETECTOR',stim.target_rec(i-self.mX-self.mZ))
        else:
            for i in range(self.mX):
                c.append_operation('DETECTOR',stim.target_rec(i-self.mX))
        

        # repeat:
        rep = stim.Circuit()
        rep.append_operation('TICK')
        rep.append_operation('DEPOLARIZE1',data_idxs)
        rep += self.cnotcircuit(pd,pg)
        rep += self.measurements(ps)
        for i in range(self.mZ+self.mX):
            rep.append_operation('DETECTOR',[stim.target_rec(i-self.mX-self.mZ),stim.target_rec(i-2*(self.mX+self.mZ))])
        if rounds > 1:
            c += (rounds-1) * rep
        

        
        # final detectors

        # final observables
        if measure_data == 'Z': 
            c.append('X_ERROR',data_idxs,ps) 
            c.append('MZ',data_idxs) 
            for det in range(self.mZ): 
                tars = self._supp_hz(det) 
                rec = [] 
                for i in tars: 
                    rec.append(i-self.n) 
                rec.append(det-self.mZ-self.mX-self.n) 
                c.append_operation('DETECTOR',[stim.target_rec(i) for i in rec]) 
            for logical in range(len(self.Lz)): 
                tars = self._supp_Lz(logical) 
                c.append_operation( "OBSERVABLE_INCLUDE", [stim.target_rec(k-self.n) for k in tars], [logical] ) 
        else: 
            c.append('Z_ERROR',data_idxs,ps) 
            c.append('MX',data_idxs) 
            for det in range(self.mX): 
                tars = self._supp_hx(det) 
                rec = [] 
                for i in tars: 
                    rec.append(i-self.n) 
                rec.append(det-self.mX-self.n) 
                c.append_operation('DETECTOR',[stim.target_rec(i) for i in rec]) 
            for logical in range(len(self.Lx)): 
                tars = self._supp_Lx(logical) 
                c.append_operation( "OBSERVABLE_INCLUDE", [stim.target_rec(k-self.n) for k in tars], [logical] )
        

        return c




class MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise:
    def __init__(self, css, noise_model='SD6', p=0.01, rounds=1,
                 custom_error_model={}, decoder='bplsd', decoder_kwargs=None):
        """
        decoder: 'pymatching' | 'bplsd'
        decoder_kwargs: only used when decoder='bplsd', passed into SinterLsdDecoder(...)
        """
        self.css = css
        self.noise_model = noise_model
        self.p = p
        self.rounds = rounds
        self.custom_error_model = custom_error_model
        self.decoder = (decoder or 'pymatching').lower()

        default_bplsd = dict(
            max_iter=5,              
            bp_method='ms',
            ms_scaling_factor=0.8,   
            schedule='parallel',      
            lsd_order=7,             
        )
        self.decoder_kwargs = default_bplsd if (decoder_kwargs is None) else {**default_bplsd, **decoder_kwargs}
        return

    @staticmethod
    def _json_safe(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.generic,)):
            return x.item()
        if isinstance(x, dict):
            return {str(k): MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise._json_safe(v)
                    for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise._json_safe(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)

    def get_error_param(self, noise_model: str, p: float, custom_error_model={}) -> dict:
        noise_model = noise_model.upper()
        if noise_model not in ['SD6', 'SI1000', 'EM3', 'CUSTOM']:
            raise ValueError(f"Unknown noise model: {noise_model}")

        if noise_model == 'SD6':
            err = {'pd': p, 'pg': p, 'ps': p, 'pp': p}
        elif noise_model == 'SI1000':
            err = {'pd': 0.1*p, 'pg': 1.0*p, 'ps': 0.5*p, 'pp': 0.5*p}
        elif noise_model == 'EM3':
            err = {'pd': 0.25*p, 'pg': 1.0*p, 'ps': 0.5*p, 'pp': 0.5*p}
        elif noise_model == 'CUSTOM':
            err = custom_error_model
        return err

    def get_sinter_task(self,
                        css,
                        mode='both',            # 'X'|'Z'
                        order='ZX',
                        noise_model='SD6',
                        p=0.01,
                        rounds=1,
                        burn_in_rounds=1,
                        custom_error_model={}):


        css_code_circuit_generator = CircuitGenerator(css)
        error_param = self.get_error_param(noise_model=noise_model, p=p, custom_error_model=custom_error_model)

        mode_up = mode.upper()
        if mode_up not in ('X', 'Z'):
            raise ValueError(f"mode must be 'X' or 'Z', got {mode!r}")

        measure_data = 'X' if mode_up == 'X' else 'Z'
        circuit = css_code_circuit_generator.generate(
            rounds=rounds,
            noise_model=error_param,
            burn_in_rounds=burn_in_rounds,
            measure_data=measure_data,   
            add_observable=True
        )

        meta = {
            'p': self.p,
            'n': getattr(css, 'n', None),
            'k': getattr(css, 'k', None),
            'hx': getattr(css, 'hx', None),
            'hz': getattr(css, 'hz', None),
            'lx': getattr(css, 'lx', None),
            'lz': getattr(css, 'lz', None),
            'rounds': rounds,
            'burn_in_rounds': burn_in_rounds,
            'mode': mode,
            'error_model': noise_model,
            'custom_error_model': custom_error_model if noise_model.lower() == 'custom' else None,
        }

        meta = self._json_safe(meta)
        yield sinter.Task(circuit=circuit, json_metadata=meta)

    def _choose_decoder(self):

        if self.decoder == 'bplsd':
            try:
                from ldpc.sinter_decoders.sinter_lsd_decoder import SinterLsdDecoder
            except Exception as ex:
                raise RuntimeError(
                    "decoder='bplsd' need ldpc package"
                ) from ex

            decoders = ['bplsd']
            custom_decoders = {
                'bplsd': SinterLsdDecoder(**self.decoder_kwargs)
            }
        else:

            decoders = ['pymatching']
            custom_decoders = None
        return decoders, custom_decoders

    def run(self, shots: int = 1_000, max_error: int = 100, num_workers: int = 4):
        decoders_ZX, custom_ZX = self._choose_decoder()


        samples_Z = sinter.collect(
            tasks=self.get_sinter_task(self.css, mode='Z', 
                                       noise_model=self.noise_model, p=self.p,
                                       rounds=self.rounds, burn_in_rounds=1,
                                       custom_error_model=self.custom_error_model),
            decoders=decoders_ZX,
            custom_decoders=custom_ZX,
            max_shots=shots//2, max_errors=max_error//2, num_workers=num_workers,
        )
        sZ = sum(getattr(r, 'shots', getattr(r, 'shot_count', 0)) for r in samples_Z)
        eZ = sum(getattr(r, 'errors', getattr(r, 'error_count', 0)) for r in samples_Z)
        if sZ == 0:
            raise RuntimeError("No shots collected for Z task.")
        pZ = eZ / sZ

        samples_X = sinter.collect(
            tasks=self.get_sinter_task(self.css, mode='X', 
                                       noise_model=self.noise_model, p=self.p,
                                       rounds=self.rounds, burn_in_rounds=1,
                                       custom_error_model=self.custom_error_model),
            decoders=decoders_ZX,
            custom_decoders=custom_ZX,
            max_shots=shots//2, max_errors=max_error//2, num_workers=num_workers,
        )
        sX = sum(getattr(r, 'shots', getattr(r, 'shot_count', 0)) for r in samples_X)
        eX = sum(getattr(r, 'errors', getattr(r, 'error_count', 0)) for r in samples_X)
        if sX == 0:
            raise RuntimeError("No shots collected for X task.")
        pX = eX / sX

        pL = 1 - (1 - pX) * (1 - pZ)
        return pL

if __name__ == '__main__':
    l = 12
    g = 6
    code_constructor = CodeConstructor(method='bb', para_dict={'l': l, 'g': g})

    gross_code = [
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0.
    ]
    code_144_8 = [0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.,
            1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1.]
    code_144_26 = [0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.,
            1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1.]
    css = code_constructor.construct(gross_code)
    # css = CodeConstructor('rotated-surface').construct(3)
    code_name = ['gross_code','bb code [144,26]','bb code [144,8]']
    i=0
    for code_ in [gross_code,code_144_26,code_144_8]:
        p_list = []
        pl_list = []
        for ppp in np.linspace(4.8,6.2,4):
            p = np.exp(-ppp)
            p_list.append(p)
            css = code_constructor.construct(code_)
            mc = MonteCarloEstimationOfLogicalErrorRateUnderCircuitLevelNoise(css,noise_model='SD6',p=p,rounds=12,custom_error_model={},decoder='bplsd')
            pl = mc.run(shots= 100_000, max_error = 1000, num_workers = 24)
            pl_list.append(pl)
            print(f'physical error rate:{p},logical error rate:{pl}')
        plt.plot(p_list,pl_list,label=f'{code_name[i]}')
        i+=1
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
