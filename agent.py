import numpy as np
from omni.isaac.kit import SimulationApp
'''Change open_usd path to your own'''
simulation_app = SimulationApp({"headless": False,  "open_usd": "C:/Users/resce/AppData/Local/ov/pkg/isaac_sim-2023.1.1/usd/reacher.usd"})

from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.dynamic_control import _dynamic_control
import time
import math
'''Change asset path to your own'''
asset_path = "C:/Users/resce/AppData/Local/ov/pkg/isaac_sim-2023.1.1/usd/reacher.usd"
simulation_context = SimulationContext(stage_units_in_meters=1.0)
add_reference_to_stage(asset_path, "/reacher")
simulation_context.initialize_physics()
simulation_context.play()
dc = _dynamic_control.acquire_dynamic_control_interface()

reward = -1
num_sweeps = 5
old_values, new_values = [0]*12, [0]*12
old_values[0], old_values[4], old_values[5], old_values[10] = -100, -100, -100, -100
new_values[0], new_values[4], new_values[5], new_values[10] = -100, -100, -100, -100
state_coordinates = [(0.3, 0.55), (0.4, 0.55), (0.5, 0.55), (0.6, 0.55),
                     (0.3, 0.45), (0.4, 0.45), (0.5, 0.45), (0.6, 0.45),
                     (0.3, 0.35), (0.4, 0.35), (0.5, 0.35), (0.6, 0.35)]
state_orientations_arr = np.array([[[0, 0], [-10.54, 1.74], [-21.73, 7.56], [-35.23, 19.9]],
                      [[-1.69, -19.5], [-11.77, -16.9], [-21.95, -11.6], [-33.08, -2.06]],
                      [[-7.74, -40.6], [-16.67, -36.2],[-25.53, -29.1], [-35.38, -18.9]]])
unavailable_states = [11, 4, 5, 10]
available_states = [x for x in np.arange(12) if x not in unavailable_states]

'''Agent training'''
for sweep in range(num_sweeps):
    for i in available_states:
        state = state_coordinates[i]
        next_up, next_down, next_left, next_right = (state[0], min(round(state[1]+0.1,2), 0.55)), (state[0], max(round(state[1]-0.1,2), 0.35)), (max(round(state[0]-0.1,2), 0.3), state[1]), (min(round(state[0]+0.1,2), 0.6), state[1])
        value_up, value_down, value_left, value_right = old_values[state_coordinates.index(next_up)], old_values[state_coordinates.index(next_down)], old_values[state_coordinates.index(next_left)], old_values[state_coordinates.index(next_right)] 
        new_values[i] = reward + max([value_up, value_down, value_left, value_right])
    old_values = new_values

'''Agent control'''
old_values_arr = np.array(old_values).reshape(3,-1)
action_arr = np.zeros_like(old_values_arr, dtype=int)
for row in range(old_values_arr.shape[0]):
    for col in range(old_values_arr.shape[1]):
        if [row,col] in [[1,0],[1,1],[2,2],[2,3]]:
            action_arr[row,col] = 5
        else:    
            value_up, value_down, value_left, value_right = old_values_arr[max(row-1,0),col], old_values_arr[min(row+1,2),col], old_values_arr[row,max(col-1,0)], old_values_arr[row,min(col+1,3)]
            action_arr[row,col] = np.argmax([value_up, value_down, value_left, value_right])

state_row, state_col = 0, 0
terminate = False
articulation = dc.get_articulation("/reacher")
dc.wake_up_articulation(articulation)
dof_ptr_1 = dc.find_articulation_dof(articulation, "RevoluteJoint_1")
dof_ptr_2 = dc.find_articulation_dof(articulation, "RevoluteJoint_2")
'''Agent render'''
while not terminate:
    action = action_arr[state_row, state_col]
    if action == 0:
        state_row = max(state_row-1,0)
    if action == 1:
        state_row = min(state_row+1,2)
    if action == 2:
        state_col = max(state_col-1,0)
    if action == 3:
        state_col = min(state_col+1,3)
    orientations = state_orientations_arr[state_row, state_col]  
    '''This is done for proper rendering'''  
    for i in range(8):
        dc.set_dof_position(dof_ptr_1, round(orientations[0]*math.pi/180,10))
        dc.set_dof_position(dof_ptr_2, round((-orientations[0]+orientations[1])*math.pi/180,10))
        simulation_context.step(render=True)
    effector = dc.get_rigid_body("/reacher/effector")
    pose = dc.get_rigid_body_pose(effector)
    print(pose.p[0], pose.p[1])
    time.sleep(2)    
    if [state_row, state_col] == [2, 3]:
        terminate = True

simulation_app.close()        

                               