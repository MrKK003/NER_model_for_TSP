import time
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def detect_outlier(data):
    # find q1 and q3 values
    q1, q3 = np.percentile(sorted(data), [25, 75])
    
    # compute IRQ
    iqr = q3 - q1
 
    # find lower and upper bounds
    upper_bound = q3 + (1.5 * iqr)
 
    outliers = [x for x in data if x >= upper_bound]
    
    return outliers

def  main():

    # New Antecedent/Consequent objects hold universe variables and membership
    # functions
    quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
    service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
    tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    quality.automf(3)
    service.automf(3)

    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
    tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
    tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])
    
    rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
    rule2 = ctrl.Rule(service['average'], tip['medium'])
    rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])
    
    tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)
    tipping.input['quality'] = 6.5
    tipping.input['service'] = 9.8

    # Crunch the numbers
    tipping.compute()
    
    print(tipping.output['tip'])





if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')