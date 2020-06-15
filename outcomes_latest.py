o = ['severe','died','discharged','alive','not hospitalized']

d_outcomes = {'critical condition, intubated as of 14.02.2020':'severe', 'death':'died', 'discharge':'discharged', nan : 0 ,
 'discharged':'disharged', 'Discharged' : 'discharged' , 'Discharged from hospital': 'discharged', 'not hospitalized': 'not hospitalized',
 'recovered' : 'discharged', 'recovering at home 03.03.2020' : 'not hospitalized',  'released from quarantine' : 'not hospitalized',
 'severe' : 'severe', 'stable' : 'alive',  'died': 'died', 'Death':'died', 'dead': 'died' ,
 'Symptoms only improved with cough. Currently hospitalized for follow-up.' : 'alive',
 'treated in an intensive care unit (14.02.2020)' : 'severe', 'Alive' : 'alive',  'Dead' : 'died' , 
 'Recovered' : 'alive',  'Stable' : 'alive' , 'Died' : 'died',  'Deceased' : 'died',  'stable condition': 'alive', 
 'Under treatment' : 'severe', 'Critical condition':'severe', 'Receiving Treatment': 'severe',
 'severe illness' : 'severe', 'unstable' : 'severe', 'critical condition':'severe', 'Hospitalized' : 'severe',
 'Migrated': 'other',  'Migrated_Other' :'other', 'https://www.mspbs.gov.py/covid-19.php' : 'other'}
