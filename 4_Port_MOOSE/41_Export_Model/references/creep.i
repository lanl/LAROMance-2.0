endtime = 1e11 # ~1500 hr
#P = 200.0e6
#T = 750.0
rhoc = 2.0e+14
dt = 1e-6
model = Zry4FY26CreepLAROMance900
date = 20260309

[Mesh]
  [rod]
    type = GeneratedMeshGenerator
    dim = 3
    xmin = 0.0
    xmax = 1.0e-3
    ymin = 0.0
    ymax = 1.0e-3
    zmin = 0.0
    zmax = 1.0e-3
    nx = 1
    ny = 1
    nz = 1
    elem_type = HEX8
  []
[]

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[AuxVariables]
  [./temperature]
  order = CONSTANT
  family = MONOMIAL             # Elemental Aux Variable "MONOMIAL"
  [../]
[]

[AuxKernels]
  [./temperature]
    variable = temperature
    type = FunctionAux
    function = temp_func
  [../]
[]

[Modules/TensorMechanics/Master]
  [all]
    strain = FINITE
    add_variables = true
    generate_output = 'strain_xx strain_yy strain_zz strain_yz strain_zx strain_xy stress_xx stress_yy stress_zz stress_yz stress_zx stress_xy vonmises_stress'
    use_automatic_differentiation = false
  []
[]

[BCs]
  [symmx]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []
  [symmy]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [symmz]
    type = DirichletBC
    variable = disp_z
    boundary = back
    value = 0
  []
  # [u_x]
  #   type = FunctionDirichletBC
  #   variable = disp_x
  #   boundary = right
  #   function = u_pos
  # []
  [pressure_x]
    type = FunctionNeumannBC
    variable = disp_x
    boundary = right
    function = pres_prof_pos
  []
[]

[Functions]
  [./pres_prof_pos]
    type = PiecewiseLinear
    x = '0 ${endtime}'
    y = '${P} ${P}'
  [../]

  # [./u_pos]
  #   type = PiecewiseLinear
  #   x = '0 ${endtime}'
  #   y = '${ustart} ${uend}'
  # [../]

  [./temp_func]
    type = PiecewiseLinear
    x = '0 ${endtime}'
    y = '${T} ${T}'
    # type = ParsedFunction
    # value = (573+t)*exp((100-abs(y))^10/1e22)
  [../]
[]

[Materials]
  [elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 180.0e9
    poissons_ratio = 0.3
  []
  [stress]
    type = ComputeMultipleInelasticStress
    inelastic_models = 'rom_stress_prediction_base'
  []
  [rom_stress_prediction_base]
    type = ${model}
    temperature = temperature
    relative_tolerance = 1e-8
    initial_cell_dislocation_density = ${rhoc} 
    max_inelastic_increment = 1e-2 
    max_relative_cell_dislocation_increment = 1e-1 
    outputs = all
    verbose = false
    stress_input_window_low_failure = EXTRAPOLATE
    stress_input_window_high_failure = EXCEPTION
    temperature_input_window_high_failure = ERROR
    temperature_input_window_low_failure = EXTRAPOLATE
    cell_input_window_low_failure = USELIMIT
    cell_input_window_high_failure = USELIMIT
    internal_solve_output_on = on_error
    internal_solve_full_iteration_history = false
    use_kdtree_indexing = true
  []
[]

[Preconditioning]
  [./smp]
    type = SMP # FDP finite difference preconditioning, SMP analytic Jacobian
    full = true
  [../]
[]


[Executioner]
  type = Transient

  solve_type = 'NEWTON'
  # petsc_options_iname = '-pc_type'
  # petsc_options_value = 'lu'
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
  petsc_options_value = 'lu       superlu_dist'

  nl_abs_tol = 1e-8
  nl_rel_tol = 1e-5
  line_search = 'none'
  automatic_scaling = true
  compute_scaling_once = true
  l_max_its = 1
  nl_max_its = 7
  end_time = ${endtime}
  nl_forced_its = 1
  [./TimeStepper]
    timestep_limiting_postprocessor = time_step_limit
    type = IterationAdaptiveDT
    optimal_iterations = 5
    iteration_window = 2
    dt = ${dt}
    growth_factor = 1.5
    cutback_factor = 0.3
  []
[]

[Postprocessors]
  [./finding_element_duration]
    type = ElementAverageMaterialProperty
    mat_prop = duration_element_search
    execute_on = 'timestep_end'
  [../]
  [./brute_force_fallback]
    type = ElementAverageMaterialProperty
    mat_prop = fallback_brute_force
    execute_on = 'timestep_end'
  [../]
  
  [./_dt]
    type = TimestepSize     # size of the time step per increment
  [../]
  [./nliterations]            # Number of iterations needed to converge timestep
    type = NumNonlinearIterations
  [../]
  [./literations]            # Number of linear iterations needed to converge timestep
    type = NumLinearIterations
  [../]
  [./numreseval]             # Number of residual evaluations
    type =  NumResidualEvaluations
  [../]
  [./residual]              # force residual after convergence (per increment)
    type = Residual
    residual_type =  final
  [../]
  [./initial_residual_before]  # force residual at initial guess (per increment) before preset?
    type = Residual
    residual_type = initial_before_preset
  [../]
  [./initial_residual_after] # force residual at initial guess (per increment) after preset?
    type = Residual
    residual_type = initial_after_preset
  [../]
  [./active_time]           # Time computer spent on simulation
    type = PerfGraphData
    section_name = "Root"
    data_type = total
  [../]
  [./creep_timestep]
    type = MaterialTimeStepPostprocessor
  [../]
  [creep_rate]
    type = ElementAverageMaterialProperty
    mat_prop = creep_rate
  []
  [./effective_strain_avg]
    type = ElementAverageValue
    variable = effective_creep_strain
  [../]
  [./vonmises_stress]
    type = ElementAverageValue
    variable = vonmises_stress
  [../]
  [./temperature]
    type = ElementAverageValue
    variable = temperature
  [../]
  [./cell_dislocations]
    type = ElementAverageValue
    variable = cell_dislocations
  [../]

  [./strain_xx]
    type = ElementAverageValue
    variable = strain_xx
  [../]
  [./strain_yy]
    type = ElementAverageValue
    variable = strain_yy
  [../]
  [./strain_zz]
    type = ElementAverageValue
    variable = strain_zz
  [../]
  [./strain_zx]
    type = ElementAverageValue
    variable = strain_zx
  [../]
  [./strain_xy]
    type = ElementAverageValue
    variable = strain_xy
  [../]
  [./strain_yz]
    type = ElementAverageValue
    variable = strain_yz
  [../]
  [./stress_zz]
    type = ElementAverageValue
    variable = stress_zz
  [../]
  [./stress_xx]
    type = ElementAverageValue
    variable = stress_xx
  [../]
  [./stress_yy]
    type = ElementAverageValue
    variable = stress_yy
  [../]
  [./stress_zx]
    type = ElementAverageValue
    variable = stress_zx
  [../]
  [./stress_xy]
    type = ElementAverageValue
    variable = stress_xy
  [../]
  [./stress_yz]
    type = ElementAverageValue
    variable = stress_yz
  [../]

  [./stress_xx_max]
    type = ElementExtremeValue
    variable = stress_xx
  [../]
  [./stress_yy_max]
    type = ElementExtremeValue
    variable = stress_yy
  [../]

  # extras
  [./time_step_limit]
    type = MaterialTimeStepPostprocessor
  [../]
[]

[Outputs]
  file_base = ${model}_1-element_T-${T}K_P-${P}MPa_rhoc-${rhoc}_${date}
  csv = true
  gnuplot = true
  print_linear_residuals = true
  perf_graph = true
  # output_on = 'initial timestep_end'
[]
