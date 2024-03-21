from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = load_dataset("ipab-rad/driving_scenarios", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

prompt  = """
Scenic is a domain-specific probabilistic programming language for generating simulations of cyber-physical systems. I want to use it to generate simulations for driving scenarios. Here is the scenic program for a scenario ""Ego vehicle goes straight at 4-way intersection adversary vehicle must stop when making the left turn to turn in the intersection to allow ego to pass.""

```
#################################
# MAP AND MODEL                 #
#################################

param map = localPath('../assets/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model

#################################
# CONSTANTS                     #
#################################

MODEL = 'vehicle.mini.cooper_s_2021'

EGO_INIT_DIST = [20, 25]
param EGO_SPEED = VerifaiRange(7, 10)

ADV_INIT_DIST = [15, 20]
param ADV_SPEED = VerifaiRange(7, 10)
param ADV_BRAKE = VerifaiRange(0.5, 1.0)

SAFE_DIST = 20
CRASH_DIST = 5
TERM_DIST = 100

#################################
# AGENT BEHAVIORS               #
#################################

behavior EgoBehavior(trajectory):
	do FollowTrajectoryBehavior(target_speed=globalParameters.EGO_SPEED, trajectory=trajectory)

behavior AdversaryBehavior(trajectory):
	try:
		do FollowTrajectoryBehavior(target_speed=globalParameters.ADV_SPEED, trajectory=trajectory)
	interrupt when withinDistanceToAnyObjs(self, SAFE_DIST):
		take SetBrakeAction(globalParameters.ADV_BRAKE)

#################################
# SPATIAL RELATIONS             #
#################################

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoInitLane.centerline

advInitLane = Uniform(*filter(lambda m:m.type is ManeuverType.STRAIGHT,egoManeuver.reverseManeuvers)).startLane
advManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, advInitLane.maneuvers))
advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
advSpawnPt = OrientedPoint in advInitLane.centerline

#################################
# SCENARIO SPECIFICATION        #
#################################

ego = Car at egoSpawnPt,
	with blueprint MODEL,
	with behavior EgoBehavior(egoTrajectory)

adversary = Car at advSpawnPt,
	with blueprint MODEL,
	with behavior AdversaryBehavior(advTrajectory)

require EGO_INIT_DIST[0] <= (distance to intersection) <= EGO_INIT_DIST[1]
require ADV_INIT_DIST[0] <= (distance from adversary to intersection) <= ADV_INIT_DIST[1]
terminate when (distance to egoSpawnPt) > TERM_DIST
```

Now generate a scenic program for a scenario "Ego vehicle goes straight at 3-way intersection adversary vehicle must stop when making the right turn to turn in the intersection to allow ego to pass."
"""

inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

# generate the response
outputs = model.generate(inputs["input_ids"], max_length=1024, do_sample=True, temperature=0.1)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)