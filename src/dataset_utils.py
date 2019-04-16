import vtk
import vtk.util.numpy_support as vutil
import numpy as np
import pandas as pd

def read_solution(path, order=None, size=10, limiting=False, advection=False):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    output = reader.GetOutput()
    n_points = reader.GetNumberOfPoints()
        
    nodes = output.GetPoints().GetData()
    po_d = output.GetPointData()
    time_idx = 3 if limiting else 1
    time = po_d.GetArray(time_idx).GetTuple(0)[0]
    
    number_of_data = int(po_d.GetArray(0).GetSize()/n_points)
    # Are we 2d or 3d?
    # Data is [rho, v, p, potT] 
    number_of_data_2d = 6 if advection else 5
    number_of_coords = 2 if number_of_data == number_of_data_2d else 3
    
    store_refinement_status = (number_of_coords == 2)
    num_refinement_cols = 1 if store_refinement_status else 0
    data = np.zeros((n_points, 3 + number_of_data + num_refinement_cols)) * float('NaN')
    for i in range(n_points):
        if number_of_coords == 2:
            x, y, z = nodes.GetTuple(i)
            if limiting:
                refinement_status = np.longfloat(output.GetPointData().GetArray(1).GetTuple(i))
            else:
                refinement_status = 4
            if advection:
                rho, v_x, v_y, p, Z, potT = np.longfloat(output.GetPointData().GetArray(0).GetTuple(i))
                data[i, :] = np.array([x, y, z, rho, v_x, v_y, p, Z, potT, refinement_status])
            else:                
                rho, v_x, v_y, p, potT = np.longfloat(output.GetPointData().GetArray(0).GetTuple(i))
                data[i, :] = np.array([x, y, z, rho, v_x, v_y, p, potT, refinement_status])
        else:
            assert(not advection and not limiting)
            x, y, z = nodes.GetTuple(i)
            rho, v_x, v_y, v_z, p, potT = np.longfloat(output.GetPointData().GetArray(0).GetTuple(i))
            data[i, :] = np.array([x, y, z, rho, v_x, v_y, v_z, p, potT])       

    if number_of_coords == 2:
        if advection:
            columns = ['x', 'y', 'z', 'rho', 'v_x', 'v_y', 'p', 'Z', 'potT', 'refinementStatus']
        else:
            columns = ['x', 'y', 'z', 'rho', 'v_x', 'v_y', 'p', 'potT', 'refinementStatus']
    else:
        columns = ['x', 'y', 'z', 'rho', 'v_x', 'v_y', 'v_z', 'p', 'potT']
    df = pd.DataFrame(data, columns=columns).dropna()

    if not order:
        # Find order from filename
        order = int(re.search(r'order_(\d)_', path).group(1))
        
    basis_size = order + 1
    dx = size/(n_points**0.5/(order+1))
    n_cells = int(10/dx)**2
    
    return df, time, order, number_of_coords, dx, n_points


def get_other_coords(coord, dim=2):
    if dim == 2:
        return ['y','z'] if coord == 'x' else ['x', 'z']
    else:
        others_map = {'x': ['y', 'z'],
                      'y': ['x', 'z'],
                      'z': ['x', 'y']}    
        return others_map[coord]
     

def cut_dataset(df, coord, dim=2, cut_at=None, every_nth=1, others=None):
    if not cut_at:
        cut_at = [None] * dim
    if not others:
        others = get_other_coords(coord, dim=dim)
        
    print(coord, others)
    df_cut = df
    for other, other_cut in zip(others, cut_at):
        if not other_cut: 
            s = sorted(df_cut[other])
            median = s[len(s)//2]
            other_cut = median # todo
        df_cut = df_cut.loc[df_cut[other] == other_cut, :].sort_values(by=coord)
        n = len(df_cut)
        #print(f"Cutting {other} at {other_cut} with {n} elements at time {time}.")    
    return df_cut.sort_values(by=coord).iloc[::every_nth,:]
       
def eval_cut(df, coord, time, mu, analytical_solution, dim=2, every_nth=5, vel_var=None):
    if not vel_var:
        vel_var_dict = {
            'x': 'u',
            'y': 'v',
            'z': 'w',
        }
        vel_var = vel_var_dict[coord]
    vel_var_approx_dict = {
        'u': 'v_x',
        'v': 'v_y',
        'w': 'v_z'
    }
    vel_var_approx = vel_var_approx_dict[vel_var]

    others = get_other_coords(coord, dim=dim)
    df_cut = cut_dataset(df, coord, dim=dim, every_nth=every_nth)
    coord_cut, other_cuts, p_approx_cut, vel_approx_cut = (df_cut[coord].values,
                                                        [df_cut[others[0]].values,
                                                        df_cut[others[1]].values],
                                                        df_cut['p'].values,
                                                        df_cut[vel_var_approx])
    # TODO 3D here

    if coord == 'x':
        p_ana_cut = analytical_solution(x=coord_cut, y=other_cuts[0], z=other_cuts[1], time=time, mu=mu, var='p')
        vel_ana_cut = analytical_solution(x=coord_cut, y=other_cuts[0], z=other_cuts[1], time=time, mu=mu, var=vel_var)
    elif coord == 'y':
        p_ana_cut = analytical_solution(y=coord_cut, x=other_cuts[0], z=other_cuts[1], time=time, mu=mu, var='p')
        vel_ana_cut = analytical_solution(y=coord_cut, x=other_cuts[0], z=other_cuts[1], time=time, mu=mu, var=vel_var)
    else:
        p_ana_cut = analytical_solution(z=coord_cut, x=other_cuts[0], y=other_cuts[1], time=time, mu=mu, var='p')
        vel_ana_cut = analytical_solution(z=coord_cut, x=other_cuts[0], y=other_cuts[1], time=time, mu=mu, var=vel_var)

    return coord_cut, p_approx_cut, p_ana_cut, vel_approx_cut, vel_ana_cut


