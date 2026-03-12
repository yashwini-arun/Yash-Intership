"""
dataset.py — Generates Disaster Response Q&A dataset
Run: python dataset.py
"""

import json, random, os

QA = [
    {
        "scenario": "6.8 magnitude earthquake. 5 buildings collapsed. 400 trapped. 6 rescue teams available.",
        "response": "1. Deploy 3 teams to densest collapse zones.\n2. Set up triage 200m away: red/yellow/green zones.\n3. Request 4 more USAR teams — 6 is insufficient for 5 buildings.\n4. Pause all ops during aftershocks.\n5. Establish command post 500m upwind.",
        "category": "earthquake"
    },
    {
        "scenario": "School building collapsed. 150 students inside. Parents gathering. 2 rescue teams on scene.",
        "response": "1. Set 200m exclusion zone immediately — crowd control is critical.\n2. Assign 1 person ONLY for parent updates every 15 mins.\n3. Use acoustic listening at 4 corners before any machinery.\n4. Check bathrooms and stairwells first — children hide there.\n5. Request 4 more teams and pediatric medical unit.",
        "category": "earthquake"
    },
    {
        "scenario": "72 hours post-collapse. Faint signals detected. Team exhausted after 3 days.",
        "response": "1. Rotate fresh team immediately — fatigue causes fatal errors.\n2. Cycle voice, tapping, and listening in 3-min silence intervals.\n3. Prepare for crush syndrome — IV fluids BEFORE extraction.\n4. Do NOT move victim rapidly — repositioning shock can be fatal.\n5. Alert hospital: hyperkalemia risk in prolonged entrapment.",
        "category": "earthquake"
    },
    {
        "scenario": "Flash flood. River rising 30cm/hr. 3 villages, 2000 residents. 4 boats. Roads submerged.",
        "response": "1. Prioritize: elderly, disabled, children — 2 boats to highest-risk village.\n2. Assembly point minimum 15m above current water level.\n3. MATH: 4 boats x 10 persons x 6 trips = 240/hr. Need 8+ hrs for 2000. REQUEST MORE BOATS NOW.\n4. Mark cleared houses with chalk.\n5. Request helicopter for rooftop rescues.",
        "category": "flood"
    },
    {
        "scenario": "Family of 5 stranded on rooftop. River current 4 knots. One team member is inexperienced swimmer.",
        "response": "1. Do NOT enter 4-knot current without swift water certification — throw rope first.\n2. Approach from downstream at 45-degree angle.\n3. Inexperienced swimmer stays IN the boat at all times.\n4. Extract children first, adult males last.\n5. Rule: a rescuer in the water becomes a second victim.",
        "category": "flood"
    },
    {
        "scenario": "Post-flood. 500 displaced in shelter. Stomach illness spreading. Latrines inadequate.",
        "response": "1. Isolate sick individuals immediately — likely cholera or typhoid.\n2. Collect stool sample from 3 patients before any treatment.\n3. Minimum ratio: 1 latrine per 20 people. Dig emergency latrines NOW.\n4. 50m exclusion zone between latrines and food/water points.\n5. Distribute ORS immediately. Chlorinate all water at 0.5mg/L.",
        "category": "flood"
    },
    {
        "scenario": "Chemical plant fire. Unknown chemicals. Black smoke. 3 workers missing. Wind toward residential area 500m away.",
        "response": "1. Do NOT enter without SCBA — black smoke means toxic compounds.\n2. Evacuate 1km radius downwind immediately.\n3. Get chemical inventory from plant safety officer before firefighting.\n4. Hot/Warm/Cold zone setup — all upwind.\n5. Do NOT apply water until chemicals identified — some (Na, Li) explode with water.",
        "category": "fire"
    },
    {
        "scenario": "High-rise fire on 14th floor. Elevators offline. 200 occupants above fire floor.",
        "response": "1. Do NOT evacuate entire building — stairwell crowding causes trampling.\n2. Evacuate floors 12-17 only. Floors 1-13: shelter in place.\n3. Floors 18+: shelter in place unless smoke detected.\n4. Pressurize stairwells via HVAC to block smoke.\n5. Rule: In high-rise fires, elevators kill. Manage stairwells.",
        "category": "fire"
    },
    {
        "scenario": "Bus accident. 45 passengers. 12 critical, 18 serious, 15 minor. 3 ambulances. Hospital 20 mins away.",
        "response": "1. START triage: RED (12 critical) load first, YELLOW (18 serious) second, GREEN (15 minor) self-evacuate.\n2. MATH: 3 ambulances x 2 critical = 6 per run. Need 2 runs for all red tags.\n3. Request 3 more ambulances and helicopter LZ for most critical.\n4. Notify hospital: 12 critical incoming — activate trauma protocol.\n5. Rule: 30 seconds max per patient during initial triage.",
        "category": "mass_casualty"
    },
    {
        "scenario": "Remote hiker injured at 3500m. Helicopter cannot fly. Team is 6 hours away. Temperature dropping to -5C tonight.",
        "response": "1. Establish radio contact — assess injuries and consciousness now.\n2. Instruct victim: insulate from ground, conserve phone battery.\n3. Send fastest 2-person team NOW with bivouac gear — reach before nightfall.\n4. Full team follows with stretcher and medical kit.\n5. At 3500m hypothermia onset is 40% faster than sea level — hot drinks on arrival.",
        "category": "search_rescue"
    },
    {
        "scenario": "Disaster camp. 5000 displaced. Food for 3 days only. Road cut off. No resupply for 7 days.",
        "response": "1. Conduct actual census — adjust rations on real count, not estimate.\n2. Implement 2/3 ration immediately — 3 days stretched to 5 at 60%.\n3. Full rations only for: children under 5, pregnant women, critically ill.\n4. Central kitchen reduces waste by 30% vs individual distribution.\n5. MATH: 5000 x 7 days x 2100 kcal = 73.5M kcal needed. Request aerial resupply NOW.",
        "category": "resource_management"
    },
    {
        "scenario": "72 hours into disaster. 3 rescue team members showing critical stress symptoms. Operations must continue.",
        "response": "1. Rotate stressed members OFF active rescue — impaired rescuers cause accidents.\n2. 10-min structured group debrief — facts only, not therapy.\n3. Mandatory 8 hours sleep in any 48-hr period — non-negotiable.\n4. Assign peer support person to each affected member.\n5. Truth: a burned-out rescue team is a liability. Protecting your team IS protecting victims.",
        "category": "resource_management"
    },
]

INSTRUCTIONS = [
    "You are an expert disaster response coordinator. Provide a prioritized action plan for this emergency.",
    "You are an experienced rescue commander. Give step-by-step response instructions for this situation.",
    "As a certified emergency management specialist, provide an immediate action plan for this scenario.",
]

def generate():
    os.makedirs("data", exist_ok=True)
    random.seed(42)
    samples = []
    for qa in QA:
        for instr in INSTRUCTIONS:
            samples.append({
                "instruction": instr,
                "input": qa["scenario"],
                "output": qa["response"],
                "category": qa["category"],
                "text": f"### Instruction:\n{instr}\n\n### Scenario:\n{qa['scenario']}\n\n### Response:\n{qa['response']}"
            })

    random.shuffle(samples)
    n = int(len(samples) * 0.85)
    train, test = samples[:n], samples[n:]

    with open("data/train.jsonl", "w") as f:
        [f.write(json.dumps(s) + "\n") for s in train]
    with open("data/test.jsonl", "w") as f:
        [f.write(json.dumps(s) + "\n") for s in test]

    print(f"✅ Dataset: {len(train)} train | {len(test)} test samples")

if __name__ == "__main__":
    generate()