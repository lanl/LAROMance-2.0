/************************************************************************************/
/* © 2026. Triad National Security, LLC. All rights reserved.                       */
/*                                                                                  */
/* This program was produced under U.S. Government contract 89233218CNA000001 for   */
/* Los Alamos National Laboratory (LANL), which is operated by Triad National       */
/* Security, LLC for the U.S. Department of Energy/National Nuclear Security        */
/* Administration. All rights in the program are reserved by Triad National         */
/* Security, LLC, and the U.S. Department of Energy/National Nuclear Security       */
/* Administration. The Government is granted for itself and others acting on its    */
/* behalf a nonexclusive, paid‑up, irrevocable worldwide license in this material to*/
/* reproduce, prepare derivative works, distribute copies to the public, perform    */
/* publicly and display publicly, and to permit others to do so.                    */
/************************************************************************************/

#include "LAROManceStressUpdateBaseUniversal.h"
#include "Function.h"
#include "MathUtils.h"
#include "MooseUtils.h"
#include "MooseRandom.h"
#include "Units.h"

#include <Eigen/Dense>
using namespace Eigen;

#include <iostream>

registerMooseObject("SolidMechanicsApp", LAROManceStressUpdateBaseUniversal);
registerMooseObject("SolidMechanicsApp", ADLAROManceStressUpdateBaseUniversal);

// Works only in -dbg mode?
template <bool is_ad>
InputParameters
LAROManceStressUpdateBaseUniversalTempl<is_ad>::validParams()
{
  InputParameters params = RadialReturnCreepStressUpdateBaseTempl<is_ad>::validParams();
  params.addClassDescription("Base class to calculate the effective creep strain based on the "
                             "rates predicted by a material specific Los Alamos Reduced Order "
                             "Model derived from a Visco-Plastic Self Consistent calculations.");

  params.addRequiredCoupledVar("temperature", "The coupled temperature (K)");
  params.addParam<MaterialPropertyName>("environmental_factor",
                                        "Optional coupled environmental factor");
  params.addParam<MaterialPropertyName>("microstruct_param_1",
                                        "Optional coupled microstructure parameter 1");
  params.addParam<MaterialPropertyName>("microstruct_param_2",
                                        "Optional coupled microstructure parameter 2");
  params.addParam<MaterialPropertyName>("microstruct_param_3",
                                        "Optional coupled microstructure parameter 3");

  MooseEnum error_lower_limit_behavior("ERROR WARN IGNORE EXCEPTION DONTHING USELIMIT",
                                       "EXCEPTION");
  // Only allow ERROR and EXCEPTION on upper bounds
  MooseEnum error_upper_limit_behavior("ERROR WARN IGNORE EXCEPTION DONTHING USELIMIT",
                                       "EXCEPTION");
  params.addParam<MooseEnum>(
      "cell_input_window_low_failure",
      error_lower_limit_behavior,
      "What to do if cell dislocation concentration is outside the lower global "
      "window of applicability.");
  params.addParam<MooseEnum>(
      "cell_input_window_high_failure",
      error_upper_limit_behavior,
      "What to do if cell dislocation concentration is outside the upper global "
      "window of applicability.");
  params.addParam<MooseEnum>("wall_input_window_low_failure",
                             error_lower_limit_behavior,
                             "What to do if wall dislocation concentration is outside the "
                             "lower global window of applicability.");
  params.addParam<MooseEnum>("wall_input_window_high_failure",
                             error_upper_limit_behavior,
                             "What to do if wall dislocation concentration is outside the "
                             "upper global window of applicability.");
  params.addParam<MooseEnum>(
      "old_strain_input_window_low_failure",
      error_lower_limit_behavior,
      "What to do if old strain is outside the lower global window of applicability.");
  params.addParam<MooseEnum>(
      "old_strain_input_window_high_failure",
      error_upper_limit_behavior,
      "What to do if old strain is outside the upper global window of applicability.");

  MooseEnum extrapolated_lower_limit_behavior(
      "ERROR WARN IGNORE EXCEPTION DONOTHING USELIMIT EXTRAPOLATE", "EXTRAPOLATE");
  params.addParam<MooseEnum>(
      "stress_input_window_low_failure",
      extrapolated_lower_limit_behavior,
      "What to do if stress is outside the lower global window of applicability.");
  params.addParam<MooseEnum>(
      "stress_input_window_high_failure",
      error_upper_limit_behavior,
      "What to do if stress is outside the upper global window of applicability.");
  params.addParam<MooseEnum>(
      "temperature_input_window_low_failure",
      extrapolated_lower_limit_behavior,
      "What to do if temperature is outside the lower global window of applicability.");
  params.addParam<MooseEnum>(
      "temperature_input_window_high_failure",
      error_upper_limit_behavior,
      "What to do if temperature is outside the upper global window of applicability.");
  params.addParam<MooseEnum>(
      "environment_input_window_low_failure",
      extrapolated_lower_limit_behavior,
      "What to do if environmental factor is outside the lower global window of applicability.");
  params.addParam<MooseEnum>(
      "environment_input_window_high_failure",
      error_upper_limit_behavior,
      "What to do if environmental factor is outside the upper global window of applicability.");

  params.addParam<MooseEnum>("microstruct_param_1_input_window_low_failure",
                             extrapolated_lower_limit_behavior,
                             "What to do if microstruct_param_1 factor is outside the lower global "
                             "window of applicability.");
  params.addParam<MooseEnum>("microstruct_param_1_input_window_high_failure",
                             error_upper_limit_behavior,
                             "What to do if microstruct_param_1 factor is outside the upper global "
                             "window of applicability.");
  params.addParam<MooseEnum>("microstruct_param_2_input_window_low_failure",
                             extrapolated_lower_limit_behavior,
                             "What to do if microstruct_param_2 factor is outside the lower global "
                             "window of applicability.");
  params.addParam<MooseEnum>("microstruct_param_2_input_window_high_failure",
                             error_upper_limit_behavior,
                             "What to do if microstruct_param_2 factor is outside the upper global "
                             "window of applicability.");
  params.addParam<MooseEnum>("microstruct_param_3_input_window_low_failure",
                             extrapolated_lower_limit_behavior,
                             "What to do if microstruct_param_3 factor is outside the lower global "
                             "window of applicability.");
  params.addParam<MooseEnum>("microstruct_param_3_input_window_high_failure",
                             error_upper_limit_behavior,
                             "What to do if microstruct_param_3 factor is outside the upper global "
                             "window of applicability.");

  params.addRangeCheckedParam<Real>(
      "initial_cell_dislocation_density",
      1e12,
      "initial_cell_dislocation_density >= 0.0",
      "Initial density of cell (glissile) dislocations (1/m^2)");
  params.addRangeCheckedParam<Real>(
      "cell_dislocations_normal_distribution_width",
      0.0,
      "cell_dislocations_normal_distribution_width >= 0.0",
      "Width of the normal distribution to assign to the initial cell dislocation value. This is "
      "given as a fraction of the initial_cell_dislocation_density.");
  params.addRangeCheckedParam<Real>(
      "max_relative_cell_dislocation_increment",
      0.5,
      "max_relative_cell_dislocation_increment > 0.0",
      "Maximum increment of density of cell (glissile) dislocations.");

  params.addRangeCheckedParam<Real>("initial_wall_dislocation_density",
                                    1e12,
                                    "initial_wall_dislocation_density >= 0.0",
                                    "wall (locked) dislocation density initial value (1/m^2).");
  params.addRangeCheckedParam<Real>(
      "wall_dislocations_normal_distribution_width",
      0.0,
      "wall_dislocations_normal_distribution_width >= 0.0",
      "Width of the normal distribution to assign to the initial wall dislocation value. This is "
      "given as a fraction of the initial_wall_dislocation_density.");
  params.addRangeCheckedParam<Real>(
      "max_relative_wall_dislocation_increment",
      0.5,
      "max_relative_wall_dislocation_increment > 0.0",
      "Maximum increment of wall (locked) dislocation density initial value (1/m^2).");

  params.addParam<bool>("verbose", false, "Flag to output verbose information.");

  params.addParam<FunctionName>(
      "cell_dislocation_density_forcing_function",
      "Optional forcing function for cell dislocation. If provided, the cell dislocation "
      "density will be reset to the function value at the beginning of the timestep. Used for "
      "testing purposes only.");
  params.addParam<FunctionName>(
      "wall_dislocation_density_forcing_function",
      "Optional forcing function for wall dislocation. If provided, the wall dislocation "
      "density will be reset to the function value at the beginning of the timestep. Used for "
      "testing purposes only.");
  params.addParam<FunctionName>(
      "old_creep_strain_forcing_function",
      "Optional forcing function for the creep strain from the previous timestep. If provided, "
      "the old creep strain will be reset to the function value at the beginning of the "
      "timestep. Used for testing purposes only.");
  params.addParam<FunctionName>(
      "effective_stress_forcing_function",
      "Optional forcing function for the effective stress. If provided, the effective stress will "
      "be reset to the function value at the beginning of the timestep. Used for testing purposes "
      "only.");

  params.addParam<bool>(
      "use_kdtree_indexing", true, "Whether to use KD-tree indexing for faster element lookups");

  params.addParam<unsigned int>("random_seed", 0, "Random number generator seed");
  params.addParam<std::string>("stress_unit", "Pa", "unit of stress");

  // use std::string here to avoid automatic absolute path expansion
  params.addParam<FileName>("model", "LaRomance model JSON datafile");
  params.addParam<FileName>("export_model", "Write LaRomance model to JSON datafile");

  params.addParamNamesToGroup(
      "cell_dislocation_density_forcing_function wall_dislocation_density_forcing_function "
      "old_creep_strain_forcing_function effective_stress_forcing_function random_seed stress_unit",
      "Advanced");

  return params;
}

template <bool is_ad>
LAROManceStressUpdateBaseUniversalTempl<is_ad>::LAROManceStressUpdateBaseUniversalTempl(
    const InputParameters & parameters)
  : RadialReturnCreepStressUpdateBaseTempl<is_ad>(parameters),
    _temperature(this->template coupledGenericValue<is_ad>("temperature")),

    // Optional uncoupled inputs
    _environmental(
        this->isParamValid("environmental_factor")
            ? &this->template getGenericMaterialProperty<Real, is_ad>("environmental_factor")
            : nullptr),
    _microstruct_param_1(
        this->isParamValid("microstruct_param_1")
            ? &this->template getGenericMaterialProperty<Real, is_ad>("microstruct_param_1")
            : nullptr),
    _microstruct_param_2(
        this->isParamValid("microstruct_param_2")
            ? &this->template getGenericMaterialProperty<Real, is_ad>("microstruct_param_2")
            : nullptr),
    _microstruct_param_3(
        this->isParamValid("microstruct_param_3")
            ? &this->template getGenericMaterialProperty<Real, is_ad>("microstruct_param_3")
            : nullptr),

    _verbose(this->template getParam<bool>("verbose")),

    
    // Optional coupled inputs for dislocation densities and stress,
    // depending on input parameters set by user in input deck
    // cell dislocations
    _use_cell_dislocations(parameters.isParamSetByUser("initial_cell_dislocation_density") ||
                           this->isParamValid("cell_dislocation_density_forcing_function")),
    _cell_dislocations(_use_cell_dislocations ? &this->template declareGenericProperty<Real, is_ad>(
                                                    this->_base_name + "cell_dislocations")
                                              : nullptr),
    _cell_dislocations_old(_use_cell_dislocations ? &this->template getMaterialPropertyOld<Real>(
                                                        this->_base_name + "cell_dislocations")
                                                  : nullptr),
    _max_cell_increment(_use_cell_dislocations ? this->template getParam<Real>(
                                                     "max_relative_cell_dislocation_increment")
                                               : 0.5),
    _cell_function(this->isParamValid("cell_dislocation_density_forcing_function")
                       ? &this->getFunction("cell_dislocation_density_forcing_function")
                       : NULL),
    _cell_dislocation_increment(0.0),

    _substepping_applied(false),

    // wall dislocations
    _use_wall_dislocations(parameters.isParamSetByUser("initial_wall_dislocation_density") ||
                           this->isParamValid("wall_dislocation_density_forcing_function")),
    _wall_dislocations(_use_wall_dislocations ? &this->template declareGenericProperty<Real, is_ad>(
                                                    this->_base_name + "wall_dislocations")
                                              : nullptr),
    _wall_dislocations_old(_use_wall_dislocations ? &this->template getMaterialPropertyOld<Real>(
                                                        this->_base_name + "wall_dislocations")
                                                  : nullptr),
    _max_wall_increment(_use_wall_dislocations ? this->template getParam<Real>(
                                                     "max_relative_wall_dislocation_increment")
                                               : 0.5),
    _wall_function(this->isParamValid("wall_dislocation_density_forcing_function")
                       ? &this->getFunction("wall_dislocation_density_forcing_function")
                       : NULL),
    _wall_dislocation_increment(0.0),

    // stress
    _stress_function(this->isParamValid("effective_stress_forcing_function")
                         ? &this->getFunction("effective_stress_forcing_function")
                         : NULL),

    _creep_strain_old_forcing_function(this->isParamValid("old_creep_strain_forcing_function")
                                           ? &this->getFunction("old_creep_strain_forcing_function")
                                           : NULL),
    _creep_rate(
        this->template declareGenericProperty<Real, is_ad>(this->_base_name + "creep_rate")),

    _cell_rate(this->isParamValid("initial_cell_dislocation_density")
                   ? &this->template declareGenericProperty<Real, is_ad>(this->_base_name +
                                                                         "cell_dislocation_rate")
                   : nullptr),
    _wall_rate(this->isParamValid("initial_wall_dislocation_density")
                   ? &this->template declareGenericProperty<Real, is_ad>(this->_base_name +
                                                                         "wall_dislocation_rate")
                   : nullptr),
    _derivative(0.0),
    _wall_dislocations_step(this->isParamValid("initial_wall_dislocation_density")
                                ? &this->template declareGenericProperty<Real, is_ad>(
                                      this->_base_name + "wall_dislocations_step")
                                : nullptr),
    _cell_dislocations_step(this->isParamValid("initial_cell_dislocation_density")
                                ? &this->template declareGenericProperty<Real, is_ad>(
                                      this->_base_name + "cell_dislocations_step")
                                : nullptr),
    _plastic_strain_increment(),

    _number_of_substeps(
        this->template declareProperty<Real>(this->_base_name + "number_of_substeps")),

    // NEW

    _duration_element_search(this->template declareGenericProperty<Real, is_ad>(
        this->_base_name + "duration_element_search")),

    _active_element(this->template declareProperty<Real>(this->_base_name + "active_element")),

    _fallback_brute_force(this->template declareGenericProperty<Real, is_ad>(
        this->_base_name + "fallback_brute_force")),
    // end NEW

    _kdtree(nullptr),
    _master_points(),
    _use_kdtree_indexing(this->template getParam<bool>("use_kdtree_indexing")

    )

{
  this->_check_range = true; // this may not be necessary?

  // load JSON datafile
  if (this->isParamValid("model"))
  {
    const auto model_file_name = this->getDataFileName("model");
    std::ifstream model_file(model_file_name.c_str());
    model_file >> _json;
  }

  setupUnitConversionFactors(parameters);
}

template <bool is_ad>
LAROManceStressUpdateBaseUniversalTempl<is_ad>::~LAROManceStressUpdateBaseUniversalTempl()
{
  delete _kdtree;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::initializeKeyMaps()
{
  // Initialize input key-to-index map with default values
  if (_cell_dislocations)
    _input_key_to_index["cell"] = _cell_input_index;
  if (_wall_dislocations)
    _input_key_to_index["wall"] = _wall_input_index;
  _input_key_to_index["stress"] = _stress_input_index;
  _input_key_to_index["old_strain"] = _old_strain_input_index;
  _input_key_to_index["temperature"] = _temperature_input_index;
  if (_environmental)
    _input_key_to_index["environmental"] = _environmental_input_index;
  if (_microstruct_param_1)
    _input_key_to_index["microstruct_param_1"] = _microstruct_param_1_input_index;
  if (_microstruct_param_2)
    _input_key_to_index["microstruct_param_2"] = _microstruct_param_2_input_index;
  if (_microstruct_param_3)
    _input_key_to_index["microstruct_param_3"] = _microstruct_param_3_input_index;

  // Initialize output key-to-index map with default values
  if (_cell_dislocations)
    _output_key_to_index["cell"] = _cell_output_index;
  if (_wall_dislocations)
    _output_key_to_index["wall"] = _wall_output_index;
  _output_key_to_index["strain"] = _strain_output_index;
}

template <bool is_ad>
unsigned int
LAROManceStressUpdateBaseUniversalTempl<is_ad>::getInputIndexByKey(const std::string & key) const
{
  auto it = _input_key_to_index.find(key);
  if (it == _input_key_to_index.end())
    mooseError("In ", _name, ": Input key '", key, "' not found in key-to-index map.");
  return it->second;
}

template <bool is_ad>
unsigned int
LAROManceStressUpdateBaseUniversalTempl<is_ad>::getOutputIndexByKey(const std::string & key) const
{
  auto it = _output_key_to_index.find(key);
  if (it == _output_key_to_index.end())
    mooseError("In ", _name, ": Output key '", key, "' not found in key-to-index map.");
  return it->second;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::setInputIndexByKey(const std::string & key,
                                                                   unsigned int index)
{
  _input_key_to_index[key] = index;

  // Update specific indices if they match the key
  if (key == "cell" && _cell_dislocations)
    _cell_input_index = index;
  else if (key == "wall" && _wall_dislocations)
    _wall_input_index = index;
  else if (key == "stress")
    _stress_input_index = index;
  else if (key == "old_strain")
    _old_strain_input_index = index;
  else if (key == "temperature")
    _temperature_input_index = index;
  else if (key == "environmental" && _environmental)
    _environmental_input_index = index;
  else if (key == "microstruct_param_1" && _microstruct_param_1)
    _microstruct_param_1_input_index = index;
  else if (key == "microstruct_param_2" && _microstruct_param_2)
    _microstruct_param_2_input_index = index;
  else if (key == "microstruct_param_3" && _microstruct_param_3)
    _microstruct_param_3_input_index = index;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::setOutputIndexByKey(const std::string & key,
                                                                    unsigned int index)
{
  _output_key_to_index[key] = index;

  // Update specific indices if they match the key
  if (key == "cell" && _cell_dislocations)
    _cell_output_index = index;
  else if (key == "wall" && _wall_dislocations)
    _wall_output_index = index;
  else if (key == "strain")
    _strain_output_index = index;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::setupUnitConversionFactors(
    const InputParameters & parameters)
{
  // Stress unit conversion factor
  const MooseUnits stress_unit_to("MPa");
  const MooseUnits stress_unit_from(parameters.get<std::string>("stress_unit"));
  _stress_ucf = stress_unit_to.convert(1, stress_unit_from);
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::exportJSON()
{
  _json["input_limits"] = getInputLimits();
  _json["nodes"] = getNodes();
  _json["connectivity_matrix"] = getConnectivityMatrix();
  _json["nodal_values"] = getNodalValues();
  _json["shapefunction_degree"] = getShapeFunctionDegree();
}

template <bool is_ad>
bool
LAROManceStressUpdateBaseUniversalTempl<is_ad>::substeppingCapabilityEnabled()
{
  return this->template getParam<bool>("use_substep");
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::initialSetup()
{

  // Initialize key-to-index maps
  initializeKeyMaps();

  // export models that are compiled in
  if (this->isParamValid("export_model"))
  {
    exportJSON();
    std::ofstream out(this->template getParam<FileName>("export_model").c_str());
    out << _json;
  }

  // Pull in relevant ROM information and perform some sanity checks
  _output_transform = getOutputTransform();
  _input_transform = getInputTransform();
  _input_limits = getInputLimits();
  _nodes = getNodes();
  _connectivity_matrix = getConnectivityMatrix();
  _nodal_values = getNodalValues();
  _extrapolation_element = getExtrapolationElement();
  _min_nodal_values = getMinValues();
  _max_nodal_values = getMaxValues();

  // resize containers to be filled later based
  // and immediately run some sanity checks:
  _num_elements = _connectivity_matrix.size();
  _num_outputs = _nodal_values.size();
  _num_inputs = _input_limits.size();

  // Set up input and output indices based on dimensionality
  _input_indices.resize(_num_inputs);
  _output_indices.resize(_num_outputs);

  // Get indices from key-to-index maps
  unsigned int idx = 0;
  // Add dislocations if available
  if (_cell_dislocations)
    _input_indices[idx++] = getInputIndexByKey("cell");
  if (_wall_dislocations)
    _input_indices[idx++] = getInputIndexByKey("wall");

  _input_indices[idx++] = getInputIndexByKey("stress");
  _input_indices[idx++] = getInputIndexByKey("old_strain");
  _input_indices[idx++] = getInputIndexByKey("temperature");

  // Add environmental factor if available
  if (_environmental)
    _input_indices[idx++] = getInputIndexByKey("environmental");

  // Add microstructure params if available
  if (_microstruct_param_1)
    _input_indices[idx++] = getInputIndexByKey("microstruct_param_1");
  if (_microstruct_param_2)
    _input_indices[idx++] = getInputIndexByKey("microstruct_param_2");
  if (_microstruct_param_3)
    _input_indices[idx++] = getInputIndexByKey("microstruct_param_3");

  // Set up output indices
  idx = 0;
  // Add dislocations output if available
  if (_cell_dislocations)
    _output_indices[idx++] = getOutputIndexByKey("cell");
  if (_wall_dislocations)
    _output_indices[idx++] = getOutputIndexByKey("wall");

  // Strain output index is always the last one
  _output_indices[idx++] = getOutputIndexByKey("strain");

  // Set up window failure handling
  _window_failure.resize(_num_inputs);

  // Cell dislocations (if applicable)
  if (_cell_dislocations)
  {
    _window_failure[getInputIndexByKey("cell")].first =
        this->isParamValid("cell_input_window_low_failure")
            ? this->template getParam<MooseEnum>("cell_input_window_low_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::USELIMIT;
    _window_failure[getInputIndexByKey("cell")].second =
        this->isParamValid("cell_input_window_high_failure")
            ? this->template getParam<MooseEnum>("cell_input_window_high_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::USELIMIT;
  }
  // Wall dislocations (if applicable)
  if (_wall_dislocations)
  {
    _window_failure[getInputIndexByKey("wall")].first =
        this->isParamValid("wall_input_window_low_failure")
            ? this->template getParam<MooseEnum>("wall_input_window_low_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::USELIMIT;
    _window_failure[getInputIndexByKey("wall")].second =
        this->isParamValid("wall_input_window_high_failure")
            ? this->template getParam<MooseEnum>("wall_input_window_high_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::USELIMIT;
  }

  // Stress
  _window_failure[_stress_input_index].first =
      this->isParamValid("stress_input_window_low_failure")
          ? this->template getParam<MooseEnum>("stress_input_window_low_failure")
                .template getEnum<WindowFailure>()
          : WindowFailure::EXTRAPOLATE;
  _window_failure[_stress_input_index].second =
      this->isParamValid("stress_input_window_high_failure")
          ? this->template getParam<MooseEnum>("stress_input_window_high_failure")
                .template getEnum<WindowFailure>()
          : WindowFailure::EXCEPTION;

  // Old strain
  _window_failure[_old_strain_input_index].first =
      this->isParamValid("old_strain_input_window_low_failure")
          ? this->template getParam<MooseEnum>("old_strain_input_window_low_failure")
                .template getEnum<WindowFailure>()
          : WindowFailure::EXCEPTION;
  _window_failure[_old_strain_input_index].second =
      this->isParamValid("old_strain_input_window_high_failure")
          ? this->template getParam<MooseEnum>("old_strain_input_window_high_failure")
                .template getEnum<WindowFailure>()
          : WindowFailure::EXCEPTION;

  // Temperature
  _window_failure[_temperature_input_index].first =
      this->isParamValid("temperature_input_window_low_failure")
          ? this->template getParam<MooseEnum>("temperature_input_window_low_failure")
                .template getEnum<WindowFailure>()
          : WindowFailure::EXTRAPOLATE;
  _window_failure[_temperature_input_index].second =
      this->isParamValid("temperature_input_window_high_failure")
          ? this->template getParam<MooseEnum>("temperature_input_window_high_failure")
                .template getEnum<WindowFailure>()
          : WindowFailure::EXCEPTION;

  // Environmental factor (if applicable)
  if (_environmental)
  {
    _window_failure[_environmental_input_index].first =
        this->isParamValid("environment_input_window_low_failure")
            ? this->template getParam<MooseEnum>("environment_input_window_low_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXTRAPOLATE;
    _window_failure[_environmental_input_index].second =
        this->isParamValid("environment_input_window_high_failure")
            ? this->template getParam<MooseEnum>("environment_input_window_high_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXCEPTION;
  }
  // Microstructure params (if applicable)
  if (_microstruct_param_1)
  {
    _window_failure[_microstruct_param_1_input_index].first =
        this->isParamValid("microstruct_param_1_input_window_low_failure")
            ? this->template getParam<MooseEnum>("microstruct_param_1_input_window_low_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXTRAPOLATE;
    _window_failure[_microstruct_param_1_input_index].second =
        this->isParamValid("microstruct_param_1_input_window_high_failure")
            ? this->template getParam<MooseEnum>("microstruct_param_1_input_window_high_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXCEPTION;
  }
  if (_microstruct_param_2)
  {
    _window_failure[_microstruct_param_2_input_index].first =
        this->isParamValid("microstruct_param_2_input_window_low_failure")
            ? this->template getParam<MooseEnum>("microstruct_param_2_input_window_low_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXTRAPOLATE;
    _window_failure[_microstruct_param_2_input_index].second =
        this->isParamValid("microstruct_param_2_input_window_high_failure")
            ? this->template getParam<MooseEnum>("microstruct_param_2_input_window_high_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXCEPTION;
  }
  if (_microstruct_param_3)
  {
    _window_failure[_microstruct_param_3_input_index].first =
        this->isParamValid("microstruct_param_3_input_window_low_failure")
            ? this->template getParam<MooseEnum>("microstruct_param_3_input_window_low_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXTRAPOLATE;
    _window_failure[_microstruct_param_3_input_index].second =
        this->isParamValid("microstruct_param_3_input_window_high_failure")
            ? this->template getParam<MooseEnum>("microstruct_param_3_input_window_high_failure")
                  .template getEnum<WindowFailure>()
            : WindowFailure::EXCEPTION;
  }

  _global_limits.resize(_num_inputs);

  // temporarily fill global limits with extreme numerical values, to later update
  for (unsigned int i = 0; i < _num_inputs; ++i)
    _global_limits[i] = {std::numeric_limits<Real>::max(), 0.0};

  // sanity checks based on sizes of containers
  bool correct_shape = true;

  if (_input_transform.size() != _num_inputs)
    correct_shape = false;

  if (_output_transform.size() != _num_outputs)
    correct_shape = false;

  if (_input_limits.size() != _num_inputs)
    correct_shape = false;

  if (!correct_shape)
    mooseError("In ", _name, ": ROM data is not the right shape.");

  // Find global limits
  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    if (_input_limits[i][0] >= _input_limits[i][1])
      mooseError("In ", _name, ": Input limits are ordered incorrectly");
    _global_limits[i].first = std::min(MetaPhysicL::raw_value(_global_limits[i].first), MetaPhysicL::raw_value(_input_limits[i][0]));
    _global_limits[i].second = std::max(MetaPhysicL::raw_value(_global_limits[i].second), MetaPhysicL::raw_value(_input_limits[i][1]));
  }

  // Prepare containers
  _input_values.resize(_num_inputs);
  _old_input_values.resize(_num_outputs);

  if (_verbose)
  {
    Moose::err << "ROM model info: " << _name << "\n";
    Moose::err << " number of elements: " << _num_elements << "\n";
    Moose::err << " degree of Lagrange shape functions: " << getShapeFunctionDegree() << "\n";
    Moose::err << " number of outputs: " << _num_outputs << "\n";
    Moose::err << " number of inputs: " << _num_inputs << "\n";
    Moose::err << " Global limits:\n";

    // Print limits for each input
    for (unsigned int i = 0; i < _num_inputs; ++i)
    {
      Moose::err << "  Input " << i << " (" << _global_limits[i].first << " - "
                 << _global_limits[i].second << ")\n";
    }

    Moose::err << std::endl;
  }

  initializeRtreeIndex();
  mooseInfo("End of initial setup.");
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::initQpStatefulProperties()
{
  MooseRandom rng;
  rng.seed(0, this->template getParam<unsigned int>("random_seed"));

  if (_cell_dislocations)
    (*_cell_dislocations)[_qp] = rng.randNormal(
        this->template getParam<Real>("initial_cell_dislocation_density"),
        this->template getParam<Real>("initial_cell_dislocation_density") *
            this->template getParam<Real>("cell_dislocations_normal_distribution_width"));

  if (_wall_dislocations)
    (*_wall_dislocations)[_qp] = rng.randNormal(
        this->template getParam<Real>("initial_wall_dislocation_density"),
        this->template getParam<Real>("initial_wall_dislocation_density") *
            this->template getParam<Real>("wall_dislocations_normal_distribution_width"));

  RadialReturnCreepStressUpdateBaseTempl<is_ad>::initQpStatefulProperties();
}


template <bool is_ad>
GenericReal<is_ad>
LAROManceStressUpdateBaseUniversalTempl<is_ad>::maximumPermissibleValue(
    const GenericReal<is_ad> & effective_trial_stress) const
{
  // Make maximum allowed scalar a little bit less than the deformation that would reduce the
  // trial stress to zero. This prevents negative trial stresses.
  return effective_trial_stress / this->_three_shear_modulus * 0.999999;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::resetIncrementalMaterialProperties()
{
  if (_cell_dislocations)
    _cell_dislocation_increment = 0.0;
  if (_wall_dislocations)
    _wall_dislocation_increment = 0.0;

  _plastic_strain_increment.zero();

  if (_wall_dislocations_step)
    (*_wall_dislocations_step)[_qp] = 0.0;
  if (_cell_dislocations_step)
    (*_cell_dislocations_step)[_qp] = 0.0;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::storeIncrementalMaterialProperties(
    const unsigned int total_number_of_substeps)
{
  if (_wall_dislocations_step)
    (*_wall_dislocations_step)[_qp] += _wall_dislocation_increment;
  if (_cell_dislocations_step)
    (*_cell_dislocations_step)[_qp] += _cell_dislocation_increment;
  _number_of_substeps[_qp] = total_number_of_substeps;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::computeStressInitialize(
    const GenericReal<is_ad> & effective_trial_stress,
    const GenericRankFourTensor<is_ad> & elasticity_tensor)
{
  RadialReturnCreepStressUpdateBaseTempl<is_ad>::computeStressInitialize(effective_trial_stress,
                                                                         elasticity_tensor);
  // Previous substep creep strain
  RankTwoTensor creep_strain_substep = this->_creep_strain_old[_qp] + _plastic_strain_increment;

  // Prepare old values
  if (_cell_dislocations)
  {
    _old_input_values[_cell_output_index] =
        _cell_function ? _cell_function->value(_t, _q_point[_qp])
                        : ((*_cell_dislocations_old)[_qp] +
                          MetaPhysicL::raw_value((*_cell_dislocations_step)[_qp]));
  }

  // Wall dislocations if applicable
  if (_wall_dislocations)
  {
    _old_input_values[_wall_output_index] =
        _wall_function ? _wall_function->value(_t, _q_point[_qp])
                       : ((*_wall_dislocations_old)[_qp] +
                          MetaPhysicL::raw_value((*_wall_dislocations_step)[_qp]));
  }

  // Strain is always the last output
  _old_input_values[_strain_output_index] =
      _creep_strain_old_forcing_function
          ? _creep_strain_old_forcing_function->value(_t, _q_point[_qp])
          : MetaPhysicL::raw_value(
                std::sqrt((creep_strain_substep).doubleContraction(creep_strain_substep) / 1.5));

  // Prepare input values
  // Cell dislocations
  if (_cell_dislocations)
    _input_values[_cell_input_index] = _old_input_values[_cell_output_index];

  // Wall dislocations if applicable
  if (_wall_dislocations)
    _input_values[_wall_input_index] = _old_input_values[_wall_output_index];

  // Stress
  _input_values[_stress_input_index] = _stress_function ? _stress_function->value(_t, _q_point[_qp])
                                                        : effective_trial_stress * 1.0e-6;

  // Old strain
  _input_values[_old_strain_input_index] = _old_input_values[_strain_output_index];

  // Temperature
  _input_values[_temperature_input_index] = _temperature[_qp];

  // Environmental factor if applicable
  if (_environmental)
    _input_values[_environmental_input_index] = (*_environmental)[_qp];
  // Microstructure params if applicable
  if (_microstruct_param_1)
    _input_values[_microstruct_param_1_input_index] = (*_microstruct_param_1)[_qp];
  if (_microstruct_param_2)
    _input_values[_microstruct_param_2_input_index] = (*_microstruct_param_2)[_qp];
  if (_microstruct_param_3)
    _input_values[_microstruct_param_3_input_index] = (*_microstruct_param_3)[_qp];
}

template <bool is_ad>
GenericReal<is_ad>
LAROManceStressUpdateBaseUniversalTempl<is_ad>::computeResidual(
    const GenericReal<is_ad> & effective_trial_stress, const GenericReal<is_ad> & scalar)
{

  mooseAssert(std::isfinite(MetaPhysicL::raw_value(effective_trial_stress)),
              "computeResidual: effective_trial_stress must be finite");
  mooseAssert(std::isfinite(MetaPhysicL::raw_value(scalar)),
              "computeResidual: scalar must be finite!");

  GenericReal<is_ad> R = 0.0;
  // GenericReal<is_ad> dR_dscalar = _derivative  we use _derivative but it shoudl really be dR_dscalar 

  // Update new stress = sigma - 3G * scalar  
  GenericReal<is_ad> trial_stress_mpa = _stress_function
                                            ? _stress_function->value(_t, _q_point[_qp])
                                            : effective_trial_stress * _stress_ucf;
  GenericReal<is_ad> dtrial_stress_dscalar = 0.0;

  // Update stress if strain is being applied, i.e. non-testing simulation
  if (this->_apply_strain)
  {
    trial_stress_mpa -= this->_three_shear_modulus * scalar * _stress_ucf;
    dtrial_stress_dscalar -= this->_three_shear_modulus * _stress_ucf;
  }

  //compute ROM strain rate 
  _input_values[_stress_input_index] = trial_stress_mpa;
  GenericReal<is_ad> dtotal_rom_effective_strain_inc_dstress = 0.0;
  auto strainrate_output = computeROM(_strain_output_index, _input_values, true);   
  GenericReal<is_ad> edot = strainrate_output.first;
  GenericReal<is_ad> total_rom_effective_strain_inc = edot * _dt;       
  
  //testing on strain rate
  if (total_rom_effective_strain_inc == std::numeric_limits<float>::infinity())
  mooseException(
      "Cutting back, backtransforming rom output provides infinite output. \nStrain rate [1/s]: ",
      total_rom_effective_strain_inc,
      "\n Strain rate [1/s]: ",
      edot);

  // Residual is the difference between the strain increment from the ROM and the scalar strain increment
  R = total_rom_effective_strain_inc - scalar; 

  // =================================================================================================
  // Derivative: dedot_disgma = dedot_dedot_mapped * dedot_mapped_dismga_mapped * dsigma_mapped_dsigma
  // =================================================================================================
  GenericReal<is_ad> dedot_mapped_dsigma_mapped = strainrate_output.second[_stress_input_index]; 
  GenericReal<is_ad> dedot_dedot_scaled = edot;
  convertOutput(dedot_dedot_scaled, _output_transform, _strain_output_index, 1, 1, true);

  // back-transform the stress input with derivative to physical space (_dt = 1)
  GenericReal<is_ad> dsigma_mapped_dsigma = trial_stress_mpa; 
  convertOutput(dsigma_mapped_dsigma, _input_transform, _stress_input_index, 1, 1, true);

  // back-transform for derivative (review for correctness)
  dtotal_rom_effective_strain_inc_dstress += dedot_dedot_scaled * dedot_mapped_dsigma_mapped  * dsigma_mapped_dsigma * _dt;

  // _derivative = dR/dscalar
  _derivative = dtotal_rom_effective_strain_inc_dstress * dtrial_stress_dscalar - 1.0;

  if (_verbose)
  {
    Moose::err << std::setprecision(18);
    GenericReal<is_ad> environmental = 0.0;
    if (_environmental)
      environmental = (*_environmental)[_qp];
    GenericReal<is_ad> microstruct_param_1 = 0.0;
    if (_microstruct_param_1)
      microstruct_param_1 = (*_microstruct_param_1)[_qp];
    GenericReal<is_ad> microstruct_param_2 = 0.0;
    if (_microstruct_param_2)
      microstruct_param_2 = (*_microstruct_param_2)[_qp];
    GenericReal<is_ad> microstruct_param_3 = 0.0;
    if (_microstruct_param_3)
      microstruct_param_3 = (*_microstruct_param_3)[_qp];
    Moose::err << "Verbose information from " << _name << ": \n";
    Moose::err << " dt: " << _dt << "\n";
    if (_cell_dislocations)
      Moose::err << " old cell disl: " << _old_input_values[_cell_output_index] << "\n";
    if (_wall_dislocations)
      Moose::err << " old wall disl: " << _old_input_values[_wall_output_index] << "\n";
    Moose::err << " initial stress (MPa): "
               << MetaPhysicL::raw_value(effective_trial_stress) * _stress_ucf << "\n";
    Moose::err << " temperature: " << MetaPhysicL::raw_value(_temperature[_qp]) << "\n";
    Moose::err << " Stress into ROM (MPa): " << _input_values[_stress_input_index]
               << "\n";
    if (_environmental)
      Moose::err << " environmental factor: " << MetaPhysicL::raw_value(environmental) << "\n";
    if (_microstruct_param_1)
      Moose::err << " microstructure param 1: " << MetaPhysicL::raw_value(microstruct_param_1)
                 << "\n";
    if (_microstruct_param_2)
      Moose::err << " microstructure param 2: " << MetaPhysicL::raw_value(microstruct_param_2)
                 << "\n";
    if (_microstruct_param_3)
      Moose::err << " microstructure param 3: " << MetaPhysicL::raw_value(microstruct_param_3)
                 << "\n";
    Moose::err << " calculated scalar strain value (subtracted from strain update increment): " << MetaPhysicL::raw_value(scalar);
    Moose::err << "\n";
    Moose::err << " old effective strain: " << _old_input_values[_strain_output_index];
    Moose::err << "\n";
    Moose::err << " effective strain increment: "
               << MetaPhysicL::raw_value(total_rom_effective_strain_inc);
    Moose::err << "\n";
    Moose::err  << " trial_stress_mpa: "       << MetaPhysicL::raw_value(trial_stress_mpa);
    Moose::err << "\n";
    Moose::err  << " dedot_dedot_scaled : "       << MetaPhysicL::raw_value(dedot_dedot_scaled);
    Moose::err << "\n";
    Moose::err  << " dedot_mapped_dsigma_mapped : "       << MetaPhysicL::raw_value(dedot_mapped_dsigma_mapped);
    Moose::err << "\n";
    Moose::err  << " dsigma_mapped_dsigma : "       << MetaPhysicL::raw_value(dsigma_mapped_dsigma);
    Moose::err << "\n";
    Moose::err  << " dtotal_rom_effective_strain_inc_dstress : "       << MetaPhysicL::raw_value(dtotal_rom_effective_strain_inc_dstress);
    Moose::err << "\n";
    Moose::err  << " _derivative: "       << _derivative;
    Moose::err << "\n";
  }

  _creep_rate[_qp] = total_rom_effective_strain_inc / _dt;


  if (!this->_apply_strain)
  {
    if (_verbose)
      Moose::err << "    Strain not applied due to apply_strain input parameter!" << std::endl;
    _derivative = 1.0;
    return 0.0;
  }

  return R;
}


template <bool is_ad>
std::pair<GenericReal<is_ad>, std::vector<GenericReal<is_ad>>>
LAROManceStressUpdateBaseUniversalTempl<is_ad>::computeROM(const unsigned out_index,
                                                          std::vector<GenericReal<is_ad>> & input_values,
                                                          const bool derivative)
{
  // Add debug timers
  static int call_count = 0;
  call_count++;
  bool do_timing = (call_count % 1000 == 0);
  auto start_total = std::chrono::high_resolution_clock::now();
  auto start_intermediate = start_total;
  auto end_intermediate = start_total;
  long long duration_element_finding = 0;

  std::vector<GenericReal<is_ad>> working_inputs = input_values;

  // // Debug: print input values
  // std::stringstream error_msg;
  // error_msg << "Debug: ";
  // for (unsigned int i = 0; i < _num_inputs; ++i)
  // {
  //   error_msg << "\nInput " << i << ": " << working_inputs[i]
  //             << ", global bounds: ["
  //             << _global_limits[i].first << ",  " << _global_limits[i].second << "]";
  // }
  // // Emit error once (avoid spamming)
  // mooseDoOnce(mooseError(error_msg.str()));

  printInBoundCheck("Check global limits within computeROM, before surrogate computation",
                    working_inputs); // timed at 0 us

  // Update the original input_values with the possibly modified working_inputs
  input_values = working_inputs;

  // initialize vector container of booleans
  std::vector<unsigned int> in_element;

  // Gather current input point in a vector and create a copy of the unscaled values
  std::vector<GenericReal<is_ad>> this_input_point(_num_inputs);
  std::vector<GenericReal<is_ad>> unscaled_input_point(_num_inputs);
  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    this_input_point[i] = working_inputs[i];
    unscaled_input_point[i] = MetaPhysicL::raw_value(this_input_point[i]);
  }

  // Scale the input directly
  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    convertOutput(this_input_point[i], _input_transform, i); // timed at 0 us
  }

  // Collect the scaled inputs as doubles
  std::vector<GenericReal<is_ad>> raw_input_point(_num_inputs);
  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    raw_input_point[i] = MetaPhysicL::raw_value(this_input_point[i]);
  }

  // Create a map of dimension names to element numbers
  std::vector<std::string> dim_names;
  std::map<std::string, int> element_numbers;
  std::vector<unsigned int> tri_dims_indices; // Instead of std::vector<std::string> tri_dims

  // Get element numbers from derived class (elem_nums are not really used, but it could be used for
  // a sanity check somewhere)
  std::vector<unsigned int> elem_nums = getElementNumbers();

  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    std::string dim_name = "dim_" + std::to_string(i);
    dim_names.push_back(dim_name);
    if (i < elem_nums.size())
      element_numbers[dim_name] = elem_nums[i];
    else
      element_numbers[dim_name] = 1; // Default to 1 element if not specified
  }

  // Get triangular dimensions from derived class
  std::vector<unsigned int> triangular_dims = getTriangularDimensions();

  // Add triangular dimensions indices directly
  for (unsigned int dim_idx : triangular_dims)
  {
    if (dim_idx < dim_names.size())
      tri_dims_indices.push_back(dim_idx);
  }

  int active_element = -1; // Initialize the variable

  // ---- helper: numerically stable barycentric  ----
  auto barycentricCoordinates = [&](const std::vector<std::vector<GenericReal<is_ad>>> & verts,
                                    const std::vector<GenericReal<is_ad>> & point,
                                    std::vector<GenericReal<is_ad>> & lambdas_out) -> bool
  {
    // verts: nverts x ndim; nverts should be ndim+1
    if (verts.empty())
      return false;

    const unsigned int nverts = verts.size();
    const unsigned int ndim = verts[0].size();
    if (nverts != ndim + 1)
      return false;

    lambdas_out.assign(nverts, 0.0);

    if (ndim == 1)
    {
      const GenericReal<is_ad> v0 = verts[0][0], v1 = verts[1][0];
      const GenericReal<is_ad> span = v1 - v0;
      if (abs(span) < 1e-14)
        return false;
      const GenericReal<is_ad> t = (point[0] - v0) / span;
      lambdas_out[0] = 1.0 - t;
      lambdas_out[1] = t;
      return true;
    }

    // T = (v1-v0, v2-v0, ...)^T  size: ndim x ndim
    DenseMatrix<GenericReal<is_ad>> T(ndim, ndim);
    for (unsigned int j = 0; j < ndim; ++j)
      for (unsigned int i = 0; i < ndim; ++i)
        T(i, j) = verts[j + 1][i] - verts[0][i];

    // rhs = point - v0
    DenseVector<GenericReal<is_ad>> rhs(ndim);
    for (unsigned int i = 0; i < ndim; ++i)
      rhs(i) = point[i] - verts[0][i];

    // Solve T * sol = rhs
    DenseVector<GenericReal<is_ad>> sol(ndim);
    if (!safe_lu_solve(T, rhs, sol))
      return false; // or mooseError("LU solve failed in computeROM");

    lambdas_out[0] = 1.0;
    for (unsigned int j = 0; j < ndim; ++j)
    {
      lambdas_out[j + 1] = sol(j);
      lambdas_out[0] -= sol(j);
    }
    return true;
  };

  {
    // Find the element containing the point
    std::vector<GenericReal<is_ad>> point_vec(raw_input_point.begin(), raw_input_point.end());
    std::vector<Eigen::MatrixXd> element_min_bounds;
    std::vector<Eigen::MatrixXd> element_max_bounds;
    std::vector<std::string> element_types;

    // Find the element: use R-tree first
    int elem_index_by_rtree = findElementUsingRtree(raw_input_point);

    if (elem_index_by_rtree > -1)
      active_element = elem_index_by_rtree;

    // Brute force fallback
    else
    {
      if (_verbose)
        mooseWarning("KD Tree not used. Continue with brute force element search..");

      for (unsigned int iel = 0; iel < _connectivity_matrix.size(); ++iel)
      {
        const auto & node_ids = _connectivity_matrix[iel];
        std::vector<std::vector<GenericReal<is_ad>>> elem_nodes;
        elem_nodes.reserve(node_ids.size());
        for (unsigned int nid : node_ids)
          elem_nodes.push_back(_nodes[nid]);

        // Determine element type based on number of nodes and dimensions
        std::string etype;
        const unsigned int num_nodes_in_element = node_ids.size();
        const unsigned int expected_simplex_nodes = _num_inputs + 1;
        const unsigned int expected_hypercube_nodes = (1u << _num_inputs); // 2^_num_inputs
        const unsigned int expected_extruded_nodes =
            (_num_inputs >= 2) ? (3u * (1u << (_num_inputs - 2))) : 0u; // tri x 2^(d-2)

        if (num_nodes_in_element == expected_simplex_nodes)
        {
          etype = "simplex";
          if (tri_dims_indices.size() != _num_inputs)
            mooseWarning("Element ",
                         iel,
                         " has ",
                         num_nodes_in_element,
                         " nodes matching a simplex, but tri_dims doesn't match.");
        }
        else if (num_nodes_in_element == expected_hypercube_nodes)
        {
          etype = "hypercube";
          if (!tri_dims_indices.empty())
            mooseWarning("Element ",
                         iel,
                         " has ",
                         num_nodes_in_element,
                         " nodes matching a hypercube, but tri_dims is not empty.");
        }
        else if (expected_extruded_nodes && num_nodes_in_element == expected_extruded_nodes)
        {
          etype = "extruded-simplex";
          if (tri_dims_indices.empty() || tri_dims_indices.size() == _num_inputs)
            mooseWarning("Element ",
                         iel,
                         " has ",
                         num_nodes_in_element,
                         " nodes matching an extruded simplex, but tri_dims doesn't match.");
        }
        else
        {
          mooseError("Element type Unknown for element number ", iel);
        }

        element_types.push_back(etype);

        // Build per-element bound vectors
        Eigen::MatrixXd min_vals(_num_inputs, 1);
        Eigen::MatrixXd max_vals(_num_inputs, 1);

        for (unsigned int d = 0; d < _num_inputs; ++d)
        {
          // Column index = element index
          min_vals(d, 0) = _min_nodal_values[d](0, iel);
          max_vals(d, 0) = _max_nodal_values[d](0, iel);
        }

        // Keep a copy for error reporting (optional, remove if memory-heavy)
        element_min_bounds.push_back(min_vals);
        element_max_bounds.push_back(max_vals);

        // Check if point is inside element
        bool inside = false;
        const GenericReal<is_ad> tol = 1e-12;

        if (etype == "hypercube")
        {
          // For hypercube, check if point is within min/max bounds
          inside = true;
          for (unsigned int d = 0; d < _num_inputs; ++d)
            if (point_vec[d] < min_vals(d, 0) - tol || point_vec[d] > max_vals(d, 0) + tol)
            {
              inside = false;
              break;
            }
        }
        else if (etype == "simplex")
        {
          // Use robust barycentric computation
          std::vector<GenericReal<is_ad>> lambda;
          bool ok = barycentricCoordinates(elem_nodes, point_vec, lambda);
          inside = ok;
          if (ok)
            for (const auto & l : lambda)
              if (MetaPhysicL::raw_value(l) < -tol)
              {
                inside = false;
                break;
              }
        }
        else if (etype == "extruded-simplex")
        {
          // Split dims into simplex (tri) and extrusion dims
          std::vector<unsigned int> simplex_indices;
          std::vector<unsigned int> extrusion_indices;
          simplex_indices.reserve(tri_dims_indices.size());
          for (unsigned int d = 0; d < _num_inputs; ++d)
          {
            if (std::find(tri_dims_indices.begin(), tri_dims_indices.end(), d) !=
                tri_dims_indices.end())
              simplex_indices.push_back(d);
            else
              extrusion_indices.push_back(d);
          }
          if (simplex_indices.size() != 2)
            mooseError("extruded-simplex requires exactly 2 simplex (triangular) dimensions.");

          // Build triangle vertex list (unique in simplex subspace)
          // We can use the first three unique simplex vertices from elem_nodes
          std::vector<std::vector<GenericReal<is_ad>>> tri_verts;
          tri_verts.reserve(3);
          const GenericReal<is_ad> vtol = 1e-14;

          auto proj_simplex = [&](const std::vector<GenericReal<is_ad>> & x)
          {
            std::vector<GenericReal<is_ad>> s(2);
            s[0] = x[simplex_indices[0]];
            s[1] = x[simplex_indices[1]];
            return s;
          };

          auto almost_equal = [&](const std::vector<GenericReal<is_ad>> & a, const std::vector<GenericReal<is_ad>> & b)
          { return (std::abs(MetaPhysicL::raw_value(a[0]) - MetaPhysicL::raw_value(b[0])) < vtol && std::abs(MetaPhysicL::raw_value(a[1]) - MetaPhysicL::raw_value(b[1])) < vtol); };

          for (const auto & nd : elem_nodes)
          {
            auto s = proj_simplex(nd);
            bool exists = false;
            for (const auto & v : tri_verts)
              if (almost_equal(s, v))
              {
                exists = true;
                break;
              }
            if (!exists)
            {
              tri_verts.push_back(s);
              if (tri_verts.size() == 3)
                break;
            }
          }
          if (tri_verts.size() != 3)
            mooseError("Failed to identify 3 triangle vertices in extruded-simplex element.");

          // point projected to simplex dims
          std::vector<GenericReal<is_ad>> p_simplex = {point_vec[simplex_indices[0]],
                                           point_vec[simplex_indices[1]]};

          // barycentric on triangle
          std::vector<GenericReal<is_ad>> lambda;
          bool ok = barycentricCoordinates(tri_verts, p_simplex, lambda);
          bool in_simplex = ok;
          if (ok)
            for (const auto & l : lambda)
              if (MetaPhysicL::raw_value(l) < -tol)
              {
                in_simplex = false;
                break;
              }

          // extrusion bounds check
          bool in_ext = true;
          for (unsigned int d : extrusion_indices)
            if (point_vec[d] < min_vals(d, 0) - tol || point_vec[d] > max_vals(d, 0) + tol)
            {
              in_ext = false;
              break;
            }

          inside = in_simplex && in_ext;
        }

        if (inside)
        {
          active_element = iel;
          break;
        }
      }
    } // end  Brute force fallback

    _active_element[_qp] =
        static_cast<Real>(active_element); // make gobal variable for postprocessing (to real)

    // Measure element finding time
    if (do_timing)
    {
      end_intermediate = std::chrono::high_resolution_clock::now();
      duration_element_finding = std::chrono::duration_cast<std::chrono::microseconds>(
                                     end_intermediate - start_intermediate)
                                     .count();
      _duration_element_search[_qp] = duration_element_finding;
      start_intermediate = end_intermediate; // Reset for next section
    }

    if (active_element == -1)
    {
      // Print input values
      std::stringstream error_msg;
      error_msg << "No element contains data point in mesh for Input values: ";
      for (unsigned int i = 0; i < _num_inputs; ++i)
      {
        error_msg << "\nInput " << i << ": " << working_inputs[i]
                  << ", scaled: " << raw_input_point[i] << ", global bounds: ["
                  << _global_limits[i].first << ",  " << _global_limits[i].second << "]";
      }
      // Emit error once (avoid spamming)
      error_msg << "\nSubstepped? " << _substepping_applied;
      mooseDoOnce(mooseError(error_msg.str()));
    }
  } // end Element Finding

  // check extrapolation element: _extrapolation_element is a vector containing "padding elements"
  // where the surrogate has not been calibrated.
  bool is_active_element_in_extrapolation =
      std::find(_extrapolation_element.begin(), _extrapolation_element.end(), active_element) !=
      _extrapolation_element.end();
  if (is_active_element_in_extrapolation)
  {
    mooseDoOnce(mooseWarning("In Extrapolation Element # ",
                             active_element,
                             " At element ",
                             this->_current_elem->id(),
                             " _qp=",
                             _qp,
                             " Coordinates ",
                             _q_point[_qp],
                             " Block=",
                             this->_current_elem->subdomain_id()));
    printInBoundCheck("In extrapolation element", working_inputs);
  }

  // compute node range for active element
  std::vector<unsigned int> node_indices = _connectivity_matrix[active_element];

  // pre-allocate containers for element node coordinates
  std::vector<std::vector<GenericReal<is_ad>>> element_coords(_num_inputs);
  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    element_coords[i].resize(node_indices.size());
  }

  // Find the nodal coordinates of the nodes in element m
  for (unsigned int i = 0; i < node_indices.size(); ++i)
  {
    for (unsigned int j = 0; j < _num_inputs; ++j)
    {
      element_coords[j][i] = _nodes[node_indices[i]][j];
    }
  }

  // find min and max value in element for each input dimension
  std::vector<GenericReal<is_ad>> min_element_values(_num_inputs);
  std::vector<GenericReal<is_ad>> max_element_values(_num_inputs);

  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    min_element_values[i] = *std::min_element(element_coords[i].begin(), element_coords[i].end());
    max_element_values[i] = *std::max_element(element_coords[i].begin(), element_coords[i].end());
  }

  // ---- determine element type again (matches logic used above) ----
  std::string active_type;
  {
    const unsigned int nn = node_indices.size();
    const unsigned int expected_simplex_nodes = _num_inputs + 1;
    const unsigned int expected_hypercube_nodes = (1u << _num_inputs);
    const unsigned int expected_extruded_nodes =
        (_num_inputs >= 2) ? (3u * (1u << (_num_inputs - 2))) : 0u;

    if (nn == expected_simplex_nodes)
      active_type = "simplex";
    else if (nn == expected_hypercube_nodes)
      active_type = "hypercube";
    else if (expected_extruded_nodes && nn == expected_extruded_nodes)
      active_type = "extruded-simplex";
    else
      mooseError("Active element has unrecognized node count: ", nn);
  }

  // compute N depending on element type
  std::vector<GenericReal<is_ad>> N;
  std::vector<unsigned int> simplex_indices, extrusion_indices;
  std::vector<std::vector<GenericReal<is_ad>>> dN_phys(_num_inputs);
  if (active_type == "hypercube")
  {
    // --------------------------------------------------------------
    // 1.  Compute isoparametric coordinates (ξ ∈ [-1, 1]) for each input
    // --------------------------------------------------------------
    std::vector<GenericReal<is_ad>> iso_inputs(_num_inputs);
    for (unsigned int i = 0; i < _num_inputs; ++i)
      iso_inputs[i] = isoparametricInput(raw_input_point[i],
                                        min_element_values[i],
                                        max_element_values[i]);

    // --------------------------------------------------------------
    // 2.  1‑D shape‑function values (derivative = false)
    // --------------------------------------------------------------
    std::vector<std::vector<GenericReal<is_ad>>> shape_functions(_num_inputs);
    for (unsigned int i = 0; i < _num_inputs; ++i)
      shape_functions[i] = iso_1d_lagrange_linear(iso_inputs[i],
                                                  /*derivative=*/false);

    // --------------------------------------------------------------
    // 3.  Outer product → N (shape‑function values)
    // --------------------------------------------------------------
    std::function<void(std::vector<GenericReal<is_ad>> &,
                      const std::vector<std::vector<GenericReal<is_ad>>> &,
                      std::vector<unsigned int> &,
                      unsigned int)>
        computeOuterProduct = [&](std::vector<GenericReal<is_ad>> & result,
                                const std::vector<std::vector<GenericReal<is_ad>>> & shapes,
                                std::vector<unsigned int> & indices,
                                unsigned int dim)
    {
      if (dim == shapes.size())
      {
        GenericReal<is_ad> product = 1.0;
        for (unsigned int i = 0; i < shapes.size(); ++i)
          product *= shapes[i][indices[i]];
        result.push_back(product);
      }
      else
      {
        for (unsigned int i = 0; i < shapes[dim].size(); ++i)
        {
          indices[dim] = i;
          computeOuterProduct(result, shapes, indices, dim + 1);
        }
      }
    };

    std::vector<unsigned int> indices(_num_inputs, 0);
    computeOuterProduct(N, shape_functions, indices, 0);   // N filled

    // --------------------------------------------------------------
    // 4.  1‑D shape‑function derivatives w.r.t. ξ (derivative = true)
    // --------------------------------------------------------------
    
    std::vector<std::vector<GenericReal<is_ad>>> dshape_functions_dxi(_num_inputs);
    for (unsigned int i = 0; i < _num_inputs; ++i)
      dshape_functions_dxi[i] = iso_1d_lagrange_linear(iso_inputs[i],
                                                      /*derivative=*/true);

    // --------------------------------------------------------------
    // 5.  Reference‑space gradient dN/dξ (one vector per dimension)
    // --------------------------------------------------------------
    std::vector<std::vector<GenericReal<is_ad>>> dN_ref(_num_inputs);
    for (unsigned int d = 0; d < _num_inputs; ++d)
    {
      // mixed set: derivative in direction d, values elsewhere
      std::vector<std::vector<GenericReal<is_ad>>> mixed(_num_inputs);
      for (unsigned int i = 0; i < _num_inputs; ++i)
        mixed[i] = (i == d) ? dshape_functions_dxi[i] : shape_functions[i];

      std::vector<GenericReal<is_ad>> dNdxi;
      computeOuterProduct(dNdxi, mixed, indices, 0);
      dN_ref[d] = std::move(dNdxi);
    }
    // --------------------------------------------------------------
    // 6.  Assemble the element Jacobian J (diag((max‑min)/2))
    // --------------------------------------------------------------
    DenseMatrix<GenericReal<is_ad>> J(_num_inputs, _num_inputs);
    J.zero();
    for (unsigned int i = 0; i < _num_inputs; ++i)
    {
      GenericReal<is_ad> span = max_element_values[i] - min_element_values[i];
      J(i, i) = span * 0.5;          // (max‑min)/2
    }

    // --------------------------------------------------------------
    // 7.  Compute J⁻¹ by solving J * x = e_k for each basis vector
    // --------------------------------------------------------------
    DenseMatrix<GenericReal<is_ad>> invJ(_num_inputs, _num_inputs);
    invJ.zero();

    for (unsigned int k = 0; k < _num_inputs; ++k)
    {
      // RHS = unit vector e_k  (DenseVector)
      DenseVector<GenericReal<is_ad>> rhs(_num_inputs);
      rhs.zero();
      rhs(k) = 1.0;

      // Solution vector (DenseVector) will receive column k of the inverse
      DenseVector<GenericReal<is_ad>> sol(_num_inputs);
      sol.zero();

      // libMesh provides lu_solve for DenseMatrix + DenseVector
      J.lu_solve(rhs, sol);   // solves J * sol = rhs

      // Store column k of invJ
      for (unsigned int i = 0; i < _num_inputs; ++i)
        invJ(i, k) = sol(i);
    }

    // --------------------------------------------------------------
    // 8.  Apply chain rule: physical gradient dN/dx = J⁻¹ * (dN/dξ)
    // --------------------------------------------------------------
    for (unsigned int p = 0; p < _num_inputs; ++p)
      dN_phys[p].resize(N.size(), GenericReal<is_ad>(0.0));

    for (unsigned int node = 0; node < N.size(); ++node)
    {
      for (unsigned int p = 0; p < _num_inputs; ++p)          // physical direction
      {
        GenericReal<is_ad> sum = 0.0;
        for (unsigned int k = 0; k < _num_inputs; ++k)      // reference direction
          sum += invJ(p, k) * dN_ref[k][node];
        dN_phys[p][node] = sum;
      }
    }
  }

  else if (active_type == "simplex")
  {
    // Build element vertices in the same order as node_indices. Slow? Move out of loop?
    std::vector<std::vector<GenericReal<is_ad>>> verts;
    verts.reserve(node_indices.size());
    for (unsigned int k = 0; k < node_indices.size(); ++k)
    {
      const auto & nd = _nodes[node_indices[k]];
      verts.push_back(std::vector<GenericReal<is_ad>>(nd.begin(), nd.begin() + _num_inputs));
    }

    // Prepare query point
    std::vector<GenericReal<is_ad>> p(_num_inputs);
    for (unsigned int i = 0; i < _num_inputs; ++i)
      p[i] = raw_input_point[i];

    // Compute barycentric coordinates
    std::vector<GenericReal<is_ad>> lambda;
    bool ok = barycentricCoordinates(verts, p, lambda);
    if (!ok)
      mooseError("Barycentric solve failed for simplex active element.");

    // Build N following
    N.resize(node_indices.size());

    // Map nodes by simplex vertex to preserve consistent ordering
    std::map<int, std::vector<unsigned int>> nodes_by_vertex;
    for (unsigned int local = 0; local < node_indices.size(); ++local)
    {
      const auto & nd = _nodes[node_indices[local]];

      int v_idx = -1;
      for (unsigned int v = 0; v < verts.size(); ++v)
      {
        const auto & vert = verts[v];
        bool equal = true;
        for (unsigned int d = 0; d < _num_inputs; ++d)
        {
          if (std::abs(MetaPhysicL::raw_value(nd[d]) - MetaPhysicL::raw_value(vert[d])) > 1e-14)
          {
            equal = false;
            break;
          }
        }
        if (equal)
        {
          v_idx = v;
          break;
        }
      }

      if (v_idx == -1)
        mooseError("Could not match node to a simplex vertex.");

      nodes_by_vertex[v_idx].push_back(local);
    }

    // Assign N in vertex order
    for (unsigned int v = 0; v < verts.size(); ++v)
    {
      if (nodes_by_vertex.find(v) == nodes_by_vertex.end())
        continue;

      for (unsigned int local : nodes_by_vertex[v])
        N[local] = lambda[v];
    }
  }
  else if (active_type == "extruded-simplex")
  {
    for (unsigned int d = 0; d < _num_inputs; ++d)
    {
      if (std::find(tri_dims_indices.begin(), tri_dims_indices.end(), d) != tri_dims_indices.end())
        simplex_indices.push_back(d);
      else
        extrusion_indices.push_back(d);
    }

    if (simplex_indices.size() != 2)
      mooseError("extruded-simplex requires exactly 2 simplex (triangular) dimensions.");

    // --- Collect unique triangle vertices in simplex subspace ---
    std::vector<std::vector<GenericReal<is_ad>>> tri_verts;
    for (unsigned int k = 0; k < node_indices.size(); ++k)
    {
      std::vector<GenericReal<is_ad>> s = {_nodes[node_indices[k]][simplex_indices[0]],
                               _nodes[node_indices[k]][simplex_indices[1]]};
      bool exists = false;
      for (auto & v : tri_verts)
        if (std::abs(MetaPhysicL::raw_value(s[0]) - MetaPhysicL::raw_value(v[0])) < 1e-14 && std::abs(MetaPhysicL::raw_value(s[1]) - MetaPhysicL::raw_value(v[1])) < 1e-14)
        {
          exists = true;
          break;
        }
      if (!exists)
        tri_verts.push_back(s);
      if (tri_verts.size() == 3)
        break;
    }

    if (tri_verts.size() != 3)
      mooseError("Failed to identify 3 triangle vertices in extruded-simplex active element.");

    // --- Compute simplex barycentric coordinates ---
    std::vector<GenericReal<is_ad>> p_simplex = {raw_input_point[simplex_indices[0]],
                                     raw_input_point[simplex_indices[1]]};
    std::vector<GenericReal<is_ad>> lambda;
    bool ok = barycentricCoordinates(tri_verts, p_simplex, lambda);
    if (!ok)
      mooseError("Barycentric solve failed for extruded-simplex.");

    // --- 1D shapes along extrusion dimensions ---
    std::map<unsigned int, std::vector<GenericReal<is_ad>>> ext_shapes;
    for (unsigned int d : extrusion_indices)
    {
      GenericReal<is_ad> iso =
          isoparametricInput(raw_input_point[d], min_element_values[d], max_element_values[d]);
      ext_shapes[d] =
          iso_1d_lagrange_linear(iso, d == _stress_input_index && derivative); // size 2
    }

    // --- Determine simplex & hypercube node counts ---
    const unsigned int n_simplex = 3; // first 3 nodes
    const unsigned int n_hypercube = node_indices.size() / n_simplex;
    const unsigned int ndims_extrusion = extrusion_indices.size();

    // --- Precompute lexicographic hypercube corners ---
    std::vector<std::vector<unsigned int>> hypercube_corners(
        n_hypercube, std::vector<unsigned int>(ndims_extrusion));

    // Old ordering
    for (unsigned int i = 0; i < n_hypercube; ++i)
    {
      unsigned int idx = i;
      for (int d = ndims_extrusion - 1; d >= 0; --d)
      {
        hypercube_corners[i][d] = idx % 2;
        idx /= 2;
      }
    }

    // --- Fill N using tensor product ---
    N.resize(node_indices.size());
    for (unsigned int s = 0; s < n_simplex; ++s)
    {
      GenericReal<is_ad> lambda_s = lambda[s]; // simplex shape function

      for (unsigned int c = 0; c < n_hypercube; ++c)
      {
        unsigned int local = s * n_hypercube + c;
        GenericReal<is_ad> val = lambda_s;

        // multiply by 1D shapes along extrusion dims
        for (unsigned int d = 0; d < ndims_extrusion; ++d)
        {
          val *= ext_shapes[extrusion_indices[d]][hypercube_corners[c][d]];
        }

        N[local] = val;
      }
    }
  }

  else
  {
    mooseError("Unhandled active element type: ", active_type);
  }

  //  -- Permute N to order anti-lexicographic
  // Works for: simplex, hypercube, extruded-simplex
  std::vector<unsigned int> perm(node_indices.size());

  // find the pure hypercube case
  bool is_hypercube = simplex_indices.empty() && extrusion_indices.empty();

  // --- Step 1: simplex vertices ---
  std::vector<std::vector<GenericReal<is_ad>>> tri_verts;
  unsigned int n_simplex = 1; // default for hypercube

  if (!simplex_indices.empty())
  {
    std::map<std::vector<GenericReal<is_ad>>, bool> seen;
    for (unsigned int k = 0; k < node_indices.size(); ++k)
    {
      std::vector<GenericReal<is_ad>> s(simplex_indices.size());
      for (unsigned int d = 0; d < simplex_indices.size(); ++d)
        s[d] = _nodes[node_indices[k]][simplex_indices[d]];

      bool exists = false;
      for (auto & v : tri_verts)
      {
        GenericReal<is_ad> dist2 = 0.0;
        for (unsigned int d = 0; d < s.size(); ++d)
          dist2 += (v[d] - s[d]) * (v[d] - s[d]);
        if (MetaPhysicL::raw_value(dist2) < 1e-14)
        {
          exists = true;
          break;
        }
      }
      if (!exists)
        tri_verts.push_back(s);
      if (tri_verts.size() == simplex_indices.size() + 1)
        break; // triangle: 3 verts
    }
    n_simplex = tri_verts.size();
  }

  // --- Step 2: hypercube / extrusion dimensions ---
  unsigned int n_hypercube = node_indices.size() / n_simplex;
  unsigned int hypercube_dims = (is_hypercube) ? _num_inputs : extrusion_indices.size();

  // Precompute hypercube corners lexicographically
  std::vector<std::vector<unsigned int>> hypercube_corners(
      n_hypercube, std::vector<unsigned int>(hypercube_dims));
  for (unsigned int i = 0; i < n_hypercube; ++i)
  {
    unsigned int idx = i;
    for (int d = hypercube_dims - 1; d >= 0; --d)
    {
      hypercube_corners[i][d] = idx % 2;
      idx /= 2;
    }
  }

  // --- Step 3: compute permutation ---
  for (unsigned int local_pos = 0; local_pos < node_indices.size(); ++local_pos)
  {
    // --- simplex index ---
    unsigned int s = 0;
    if (!simplex_indices.empty())
    {
      std::vector<GenericReal<is_ad>> proj(simplex_indices.size());
      for (unsigned int d = 0; d < simplex_indices.size(); ++d)
        proj[d] = _nodes[node_indices[local_pos]][simplex_indices[d]];

      GenericReal<is_ad> min_dist = 1e20;
      for (unsigned int v = 0; v < tri_verts.size(); ++v)
      {
        GenericReal<is_ad> dist = 0.0;
        for (unsigned int d = 0; d < proj.size(); ++d)
          dist += (proj[d] - tri_verts[v][d]) * (proj[d] - tri_verts[v][d]);
        if (dist < min_dist)
        {
          min_dist = dist;
          s = v;
        }
      }
    }

    // --- hypercube / extrusion index ---
    unsigned int c = 0;
    if (hypercube_dims > 0)
    {
      std::vector<unsigned int> bits(hypercube_dims);
      for (unsigned int d = 0; d < hypercube_dims; ++d)
      {
        GenericReal<is_ad> val;
        if (is_hypercube)
          val = _nodes[node_indices[local_pos]][d]; // all dims
        else
          val = _nodes[node_indices[local_pos]][extrusion_indices[d]]; // extruded dims

        GenericReal<is_ad> mid = 0.5 * (min_element_values[is_hypercube ? d : extrusion_indices[d]] +
                            max_element_values[is_hypercube ? d : extrusion_indices[d]]);
        bits[d] = (val > mid) ? 1 : 0;
      }

      // find matching corner
      for (unsigned int corner = 0; corner < n_hypercube; ++corner)
      {
        bool match = true;
        for (unsigned int d = 0; d < hypercube_dims; ++d)
          if (bits[d] != hypercube_corners[corner][d])
          {
            match = false;
            break;
          }
        if (match)
        {
          c = corner;
          break;
        }
      }
    }

    perm[local_pos] = s * n_hypercube + c;
  }

  // --- Step 4: apply permutation ---
  std::vector<GenericReal<is_ad>> N_perm(node_indices.size());
  for (unsigned int i = 0; i < node_indices.size(); ++i)
    N_perm[i] = N[perm[i]];

  N.swap(N_perm);


  // repeat permutation for dN_phys:
  for (unsigned int p = 0; p < _num_inputs; ++p)
  {
      std::vector<GenericReal<is_ad>> tmp(node_indices.size());
      for (unsigned int i = 0; i < node_indices.size(); ++i)
          tmp[i] = dN_phys[p][perm[i]];
      dN_phys[p].swap(tmp);
  }

  // ----------------------------------------------------------
  // ---- End: Permute N for anti-lexicogrtaphic ordering ----
  // ----------------------------------------------------------

  // Extract the nodal values of the element indices
  std::vector<GenericReal<is_ad>> nodal_values;
  nodal_values.reserve(node_indices.size());
  for (unsigned int index : node_indices)
    nodal_values.push_back(_nodal_values[out_index][index]);

  // Check if both arrays have the same size
  if (N.size() != nodal_values.size())
    mooseError("Arrays N and nodal_values must have the same size. N.size() = ",
               N.size(),
               ", nodal_values.size() = ",
               nodal_values.size());

  // compute the rom rate output (manual dot product between N and nodal_values)
  GenericReal<is_ad> y = 0.0;
  for (size_t i = 0; i < N.size(); ++i)
    y += N[i] * nodal_values[i];
  
  // convert the output back to physical units, with dt = 1 so that the rate, and not the incremental update, is computed
  convertOutput(y, _output_transform, out_index, 1, 1, false);

  // compute the rom rate derivative output (manual dot product between dN_phys and nodal_values)
  std::vector<GenericReal<is_ad>> doutput_scaled_dx_scaled;
  doutput_scaled_dx_scaled.resize(_num_inputs);
  for (unsigned int p = 0; p < _num_inputs; ++p)          // loop over spatial dimensions
  {
      GenericReal<is_ad> sum = 0.0;
      for (size_t n = 0; n < dN_phys[p].size(); ++n)    // loop over element nodes
          sum += dN_phys[p][n] * nodal_values[n];       // ∂N_n/∂x_p * u_n
      doutput_scaled_dx_scaled[p] = sum;                                 // store component p
  }

  // return y is not back-transformed to physical space yet.
  return std::make_pair(y, std::move(doutput_scaled_dx_scaled));
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::printInBoundCheck(
    const std::string & inputString, std::vector<GenericReal<is_ad>> & input_values)
{
  // Check each input against its global limits
  for (unsigned int i = 0; i < _num_inputs; ++i)
  {
    if (!checkInBounds(input_values[i], _global_limits[i].first, _global_limits[i].second))
    {
      // Handle lower bound violations
      if (input_values[i] < _global_limits[i].first)
      {
        switch (_window_failure[i].first)
        {
          case WindowFailure::ERROR:
            mooseError("ERROR: Input ",
                       i,
                       " below bounds: ",
                       input_values[i],
                       ". Lower limit: ",
                       _global_limits[i].first,
                       "\n\tPrinted in: ",
                       inputString);
            break;
          case WindowFailure::WARN:
            mooseWarning("WARNING: Input ",
                         i,
                         " below bounds: ",
                         input_values[i],
                         ". Lower limit: ",
                         _global_limits[i].first,
                         "\n\tPrinted in: ",
                         inputString);
            break;
          case WindowFailure::EXCEPTION:
            mooseException("EXCEPTION: Input ",
                           i,
                           " below bounds: ",
                           input_values[i],
                           ". Lower limit: ",
                           _global_limits[i].first,
                           "\n\tPrinted in: ",
                           inputString);
            break;
          case WindowFailure::EXTRAPOLATE:
            // TOADD: For now do nothing
            break;
          case WindowFailure::USELIMIT:
          {
            auto temp_original_input = input_values[i];
            input_values[i] = _global_limits[i].first; // Actually modify the value
            mooseWarning("Input ",
                         i,
                         " below bounds: ",
                         temp_original_input,
                         ". Set to lower limit: ",
                         _global_limits[i].first,
                         "\n\tPrinted in: ",
                         inputString);
          }
          break;

            break;
          default:
            // IGNORE, DONOTHING - do nothing
            break;
        }
      }
      // Handle upper bound violations
      else if (input_values[i] > _global_limits[i].second)
      {
        switch (_window_failure[i].second)
        {
          case WindowFailure::ERROR:
            mooseError("ERROR: Input ",
                       i,
                       " above bounds: ",
                       input_values[i],
                       ". Upper limit: ",
                       _global_limits[i].second,
                       "\n\tPrinted in: ",
                       inputString);
            break;
          case WindowFailure::WARN:
            mooseWarning("WARNING: Input ",
                         i,
                         " above bounds: ",
                         input_values[i],
                         ". Upper limit: ",
                         _global_limits[i].second,
                         "\n\tPrinted in: ",
                         inputString);
            break;
          case WindowFailure::EXCEPTION:
            mooseException("EXCEPTION: Input ",
                           i,
                           " above bounds: ",
                           input_values[i],
                           ". Upper limit: ",
                           _global_limits[i].second,
                           "\n\tPrinted in: ",
                           inputString);
            break;
          case WindowFailure::EXTRAPOLATE:
            // TOADD: For now do nothing
            break;
          case WindowFailure::USELIMIT:
          {
            auto temp_original_input = input_values[i];
            input_values[i] = _global_limits[i].first; // Actually modify the value
            mooseWarning("Input ",
                         i,
                         " above bounds: ",
                         temp_original_input,
                         ". Set to upper limit: ",
                         _global_limits[i].second,
                         "\n\tPrinted in: ",
                         inputString);
          }
          break;
          default:
            // IGNORE, DONOTHING - do nothing
            break;
        }
      }
    }
  }
}

template <bool is_ad>
bool
LAROManceStressUpdateBaseUniversalTempl<is_ad>::checkInBounds(const GenericReal<is_ad> & input,
                                                              const GenericReal<is_ad> bound_min,
                                                              const GenericReal<is_ad> bound_max)
{
  if (input < bound_min || input > bound_max)
    return false;
  return true;
}

template <bool is_ad>
GenericReal<is_ad>
LAROManceStressUpdateBaseUniversalTempl<is_ad>::isoparametricInput(
    const GenericReal<is_ad> & input,
    const GenericReal<is_ad> node_min,
    const GenericReal<is_ad> node_max)
{
  // input
  GenericReal<is_ad> x = input;

  // isoparametric coordinate between -1.0 and 1.0
  return 2.0 * (x - node_min) / (node_max - node_min) - 1.0;
}

template <bool is_ad>
std::vector<GenericReal<is_ad>>
LAROManceStressUpdateBaseUniversalTempl<is_ad>::iso_1d_lagrange_linear(
    const GenericReal<is_ad> iso_input, const bool derivative)
{
  std::vector<GenericReal<is_ad>> Nzeta(2);
  if (derivative)
  {
    Nzeta[0] = -0.5;
    Nzeta[1] = 0.5;
  }
  else
  {
    Nzeta[0] = 0.5 * (1 - iso_input);
    Nzeta[1] = 0.5 * (1 + iso_input);
  }
  return Nzeta;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::computeStressFinalize(
    const GenericRankTwoTensor<is_ad> & plastic_strain_increment)
{
  _substepping_applied = false;
  if (_cell_dislocations)
    _cell_dislocation_increment = 0.0;
  if (_wall_dislocations)
    _wall_dislocation_increment = 0.0;

  if (_input_values[_stress_input_index])
  {
    // Without substepping
    if (_cell_dislocations)
    {
      auto cell_output = computeROM(_cell_output_index, _input_values, false);
      _cell_dislocation_increment += cell_output.first * _dt;
    }

    if (_wall_dislocations)
    {
      auto wall_output = computeROM(_wall_output_index, _input_values, false);
      _wall_dislocation_increment += wall_output.first * _dt;
    }
  }
  if (_cell_dislocations)
  {
    (*_cell_rate)[_qp] = _cell_dislocation_increment / _dt;
    (*_cell_dislocations)[_qp] = _old_input_values[_cell_output_index] + _cell_dislocation_increment;
  }

  if (_wall_dislocations)
  {
    (*_wall_rate)[_qp] = _wall_dislocation_increment / _dt;
    (*_wall_dislocations)[_qp] =
        _old_input_values[_wall_output_index] + _wall_dislocation_increment;
  }

  // Final bounds check to correct any overshoots immediately (USELIMIT, EXCPEPTION)
  std::vector<GenericReal<is_ad>> corrected_inputs = _input_values;
  corrected_inputs.resize(_num_inputs);
  if (_cell_dislocations)
    corrected_inputs[_cell_input_index] = (*_cell_dislocations)[_qp];
  if (_wall_dislocations)
    corrected_inputs[_wall_input_index] = (*_wall_dislocations)[_qp];

  printInBoundCheck("Final bounds check in computeStressFinalize", corrected_inputs);

  // Update values if they were modified by the bounds check (also increments are updated
  // accordingly)
  if (_cell_dislocations)
  {
    (*_cell_dislocations)[_qp] = corrected_inputs[_cell_input_index];
    _cell_dislocation_increment =
        corrected_inputs[_cell_input_index] - _old_input_values[_cell_output_index];
  }
  if (_wall_dislocations)
  {
    (*_wall_dislocations)[_qp] = corrected_inputs[_wall_input_index];
    _wall_dislocation_increment =
        corrected_inputs[_wall_input_index] - _old_input_values[_wall_output_index];
  }

  // For (possibly) substepping.
  _plastic_strain_increment += MetaPhysicL::raw_value(plastic_strain_increment);

  if (_verbose)
  { 
    Moose::err << std::setprecision(18);
    Moose::err << " Finalized ROM output\n";
     Moose::err << " trial stress into ROM (MPa): "
               << MetaPhysicL::raw_value(_input_values[_stress_input_index])<< "\n";
    Moose::err << " temperature: " << MetaPhysicL::raw_value(_temperature[_qp]) << "\n";
    Moose::err << "  effective creep strain increment: "
               << std::sqrt(2.0 / 3.0 *
                            MetaPhysicL::raw_value(_plastic_strain_increment.doubleContraction(
                                _plastic_strain_increment)))
               << "\n";
    Moose::err << "  total effective creep strain: "
               << std::sqrt(2.0 / 3.0 *
                            MetaPhysicL::raw_value(this->_creep_strain[_qp].doubleContraction(
                                this->_creep_strain[_qp])))
               << "\n";
    Moose::err << "  creep rate: " << MetaPhysicL::raw_value(_creep_rate[_qp]) << "\n";
    if (_cell_dislocations)
    {
      Moose::err << "  cell dislocation rate: " << MetaPhysicL::raw_value((*_cell_rate)[_qp]) << "\n";
      Moose::err << "  old cell dislocations: "
                 << MetaPhysicL::raw_value(_old_input_values[_cell_output_index]) << "\n";
      Moose::err << "  new cell dislocations: " << MetaPhysicL::raw_value((*_cell_dislocations)[_qp])
                 << "\n";
    }
    if (_wall_dislocations)
    {
      Moose::err << "  wall dislocation rate: " << MetaPhysicL::raw_value((*_wall_rate)[_qp])
                 << "\n";
      Moose::err << "  old wall dislocations: "
                 << MetaPhysicL::raw_value(_old_input_values[_wall_output_index]) << "\n";
      Moose::err << "  new wall dislocations: "
                 << MetaPhysicL::raw_value((*_wall_dislocations)[_qp]) << "\n";
    }
    Moose::err << std::endl;
  }

  RadialReturnCreepStressUpdateBaseTempl<is_ad>::computeStressFinalize(
      MetaPhysicL::raw_value(_plastic_strain_increment));
}

template <bool is_ad>
Real
LAROManceStressUpdateBaseUniversalTempl<is_ad>::computeTimeStepLimit()
{
  Real limited_dt = RadialReturnCreepStressUpdateBaseTempl<is_ad>::computeTimeStepLimit();

  // limit dt based on strain increment
  if (limited_dt < _dt && _verbose)
  {
    mooseWarning("\033[1;31m\nTime step limited by Strain increment."
                 "\ndt: ",
                 _dt,
                 "\nlimited_dt: ",
                 limited_dt,
                 "\033[0m"); // reset color
  }

  // limit based on cell dislocation increment
  // if (!_substepping_applied)
  if (_cell_dislocations)
  {
    // Real just used locally for conditional check.
    Real cell_dislocation_inc = std::abs(MetaPhysicL::raw_value(_cell_dislocation_increment));
    Real old_cell = std::abs(MetaPhysicL::raw_value(_old_input_values[_cell_output_index]));
    Real cell_candidate_dt = _dt; // Default to current dt

    // DEBUG prints
    if (old_cell == 0.0)
      mooseError("old_cell = zero");

    // Handle infinite or very large dislocation increment
    if (std::isinf(cell_dislocation_inc) || cell_dislocation_inc > 1e20)
    {
      mooseWarning(
          "\033[1;31m\nInfinite or extremely large (> 1e20) cell dislocation increment detected."
          "\nCutting time step in half."
          "\ndt: ",
          _dt,
          "\nnew dt: ",
          _dt * 0.5,
          "\033[0m");
      cell_candidate_dt = _dt * 0.5; // Cut time step in half
    }
    else
    {
      // Normal calculation when not infinite
      cell_candidate_dt = _dt * _max_cell_increment * old_cell / cell_dislocation_inc;

      if (cell_candidate_dt == 0.0)
        mooseError("cell_candidate_dt = zero");
    }

    // Debug print
    if (cell_candidate_dt < limited_dt && cell_candidate_dt < _dt && _verbose)
    {
      mooseWarning("\033[1;35m\nTime step limited by Cell dislocation increment."
                   "\noriginal _dt: ",
                   _dt,
                   "\nlimited by strain increment 'limited_dt': ",
                   limited_dt,
                   "\ncell_candidate_dt: ",
                   cell_candidate_dt,
                   "\ncell disl input: ",
                   _input_values[_cell_input_index],
                   "\ncell disl. rate: ",
                   _cell_dislocation_increment / _dt,
                   "\033[0m");
    }

    limited_dt = std::min(limited_dt, cell_candidate_dt);

    // Debug info
    if (limited_dt <= 1e-12)
    {
      mooseWarning("DEBUG: \ncell dislocation increment = ",
                   cell_dislocation_inc,
                   ", old cell disclocation = ",
                   old_cell,
                   ", original _dt = ",
                   _dt,
                   ", _max_cell_increment = ",
                   _max_cell_increment,
                   ", candidate limited_dt = ",
                   cell_candidate_dt,
                   ", new limited_dt = ",
                   limited_dt);
    }
  }

  if (_wall_dislocations)
  {
    Real wall_dislocation_inc = std::abs(MetaPhysicL::raw_value(_wall_dislocation_increment));
    Real old_wall = std::abs(MetaPhysicL::raw_value(_old_input_values[_wall_output_index]));
    Real wall_candidate_dt = _dt * _max_wall_increment * old_wall / wall_dislocation_inc;

    // Debug print
    if (wall_candidate_dt < limited_dt && wall_candidate_dt < _dt && _verbose)
    {
      mooseWarning("\nTime step limited by Wall dislocation increment");
    }

    limited_dt = std::min(limited_dt, wall_candidate_dt);
  }

  // Ensure dt never goes negative or too small
  if (limited_dt <= 0.0)
  {
    mooseWarning("Computed limited_dt <= 0; resetting to minimum allowed value.\n"
                 "\tdt = ",
                 limited_dt);
    limited_dt = 1e-14;
  }
  return limited_dt;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::outputIterationSummary(
    std::stringstream * iter_output, const unsigned int total_it)
{
  if (iter_output)
  {
    *iter_output << "At element " << this->_current_elem->id() << " _qp=" << _qp << " Coordinates "
                 << _q_point[_qp] << " block=" << this->_current_elem->subdomain_id() << '\n';
    if (_cell_dislocations)
      *iter_output << " dt " << _dt << " old cell disl: " << _old_input_values[_cell_output_index];

    if (_wall_dislocations)
      *iter_output << " old wall disl: " << _old_input_values[_wall_output_index];

    *iter_output << " old effective strain: " << _old_input_values[_strain_output_index] << "\n";
    *iter_output << " temp: " << MetaPhysicL::raw_value(_temperature[_qp]);

    if (_environmental)
      *iter_output << " environmental: " << MetaPhysicL::raw_value((*_environmental)[_qp]);
    if (_microstruct_param_1)
      *iter_output << " microstruct_param_1: "
                   << MetaPhysicL::raw_value((*_microstruct_param_1)[_qp]);
    if (_microstruct_param_2)
      *iter_output << " microstruct_param_2: "
                   << MetaPhysicL::raw_value((*_microstruct_param_2)[_qp]);
    if (_microstruct_param_3)
      *iter_output << " microstruct_param_3: "
                   << MetaPhysicL::raw_value((*_microstruct_param_3)[_qp]);

    *iter_output << " trial stress into rom (MPa): "
                 << MetaPhysicL::raw_value(_input_values[_stress_input_index]);
    if (_cell_dislocations)
      *iter_output << " cell: " << MetaPhysicL::raw_value(_input_values[_cell_input_index]);

    if (_wall_dislocations)
      *iter_output << " wall: " << MetaPhysicL::raw_value(_input_values[_wall_input_index]);

    *iter_output << " old strain: "
                 << MetaPhysicL::raw_value(_input_values[_old_strain_input_index]) << "\n";
    *iter_output << "\n";
  }
  SingleVariableReturnMappingSolutionTempl<is_ad>::outputIterationSummary(iter_output, total_it);
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::outputIterationStep(
    std::stringstream * iter_output,
    const GenericReal<is_ad> & effective_trial_stress,
    const GenericReal<is_ad> & scalar,
    const Real reference_residual)
{
  SingleVariableReturnMappingSolutionTempl<is_ad>::outputIterationStep(
      iter_output, effective_trial_stress, scalar, reference_residual);
  if (iter_output)
    *iter_output << " derivative: "
                 << MetaPhysicL::raw_value(computeDerivative(effective_trial_stress, scalar))
                 << std::endl;
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::checkJSONKey(const std::string & key)
{
  if (!this->isParamValid("model"))
    this->paramError("model", "Specify a JSON data filename.");

  const auto model_file_name = this->_pars.rawParamVal("model");
  if (_json.empty())
    this->paramError("model", "The specified JSON data file '", model_file_name, "' is empty.");
  if (!_json.contains(key))
    this->paramError(
        "model", "The key '", key, "' is missing from the JSON data file '", model_file_name, "'.");
}

template <bool is_ad>
void
LAROManceStressUpdateBaseUniversalTempl<is_ad>::initializeRtreeIndex()
{
  if (_verbose)
    mooseInfo("initializeRtreeIndex started");

  if (!_use_kdtree_indexing)
  {
    mooseInfo("kdtree indexing disabled");
    return;
  }

  mooseInfo("Building ", _num_elements, " centroids");

  _master_points.resize(_num_elements);

  for (unsigned int e = 0; e < _num_elements; ++e)
  {
    const auto & conn = _connectivity_matrix[e];
    if (conn.empty())
      mooseError("Empty conn elem ", e);

    std::vector<GenericReal<is_ad>> centroid(_num_inputs, 0.0);
    for (const auto & nid : conn)
    {
      const auto & node = _nodes[nid];
      if (node.size() != _num_inputs)
        mooseError("Dim mismatch elem ", e);
      for (unsigned int d = 0; d < _num_inputs; ++d)
        centroid[d] += node[d];
    }
    for (auto & v : centroid)
      v /= conn.size();

    _master_points[e] = centroid;
  }

  // Convert to Real for k-d tree (now stored as member variable)
  _master_points_real.resize(_num_elements);
  for (unsigned int e = 0; e < _num_elements; ++e)
  {
    _master_points_real[e].resize(_num_inputs);
    for (unsigned int d = 0; d < _num_inputs; ++d)
      _master_points_real[e][d] = MetaPhysicL::raw_value(_master_points[e][d]);
  }

  // Create adaptor with persistent Real values
  _adaptor = std::make_unique<ROMAdaptor>(_master_points_real);

  // Pass reference to the persistent adaptor
  _kdtree = new KDTreeAdaptor(_num_inputs,
                              *_adaptor,
                              nanoflann::KDTreeSingleIndexAdaptorParams(10));
  _kdtree->buildIndex();

  mooseInfo("KDTree built successfully");
}


template <bool is_ad>
int
LAROManceStressUpdateBaseUniversalTempl<is_ad>::findElementUsingRtree(
    const std::vector<GenericReal<is_ad>> & point) const
{
  if (!_kdtree || !_use_kdtree_indexing)
  {
    _fallback_brute_force[_qp] = 1.0;
    return -1;
  }

  // Extract raw values for k-d tree search (geometry only, no AD needed)
  std::vector<Real> point_raw(point.size());
  for (size_t i = 0; i < point.size(); ++i)
    point_raw[i] = MetaPhysicL::raw_value(point[i]);

  unsigned int indices_out[10];
  Real dists_out[10];
  size_t num_results = _kdtree->knnSearch(
      point_raw.data(),  // Use raw values
      10,
      indices_out,
      dists_out);

  if (num_results == 0)
  {
    _fallback_brute_force[_qp] = 1.0;
    return -1;
  }

  const GenericReal<is_ad> tol = 1e-12;
  for (size_t i = 0; i < num_results; ++i)
  {
    unsigned int elem = indices_out[i];
    bool inside = true;
    for (unsigned int d = 0; d < _num_inputs; ++d)
    {
      const Real min_val = _min_nodal_values[d](0, elem);
      const Real max_val = _max_nodal_values[d](0, elem);
      if (point[d] < min_val - tol || point[d] > max_val + tol)
      {
        inside = false;
        break;
      }
    }
    if (inside)
    {
      _fallback_brute_force[_qp] = 0.0;
      return static_cast<int>(elem);
    }
  }
  _fallback_brute_force[_qp] = 1.0;
  return -1;
}

template class LAROManceStressUpdateBaseUniversalTempl<false>;
template class LAROManceStressUpdateBaseUniversalTempl<true>;
