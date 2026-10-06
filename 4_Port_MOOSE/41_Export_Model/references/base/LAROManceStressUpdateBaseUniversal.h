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


#pragma once

#include "RadialReturnCreepStressUpdateBase.h"
#include "nlohmann/json.h"
#include "libmesh/nanoflann.hpp"

struct ROMAdaptor
{
  const std::vector<std::vector<Real>> & pts;
  ROMAdaptor(const std::vector<std::vector<Real>> & p) : pts(p) {}
  size_t kdtree_get_point_count() const { return pts.size(); }
  Real kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx][dim]; }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX &) const
  {
    return false;
  }
};

using KDTreeAdaptor =
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<Real, ROMAdaptor>,
                                        ROMAdaptor,
                                        -1 /* dynamic dim */>;

template <bool is_ad>
class LAROManceStressUpdateBaseUniversalTempl : public RadialReturnCreepStressUpdateBaseTempl<is_ad>
{

public:
  static InputParameters validParams();

  LAROManceStressUpdateBaseUniversalTempl(const InputParameters & parameters);

  virtual ~LAROManceStressUpdateBaseUniversalTempl();

  virtual void resetIncrementalMaterialProperties() override;
  virtual void
  storeIncrementalMaterialProperties(const unsigned int total_number_substeps) override;

protected:
  virtual void exportJSON();

  virtual bool substeppingCapabilityEnabled() override;

  virtual void initialSetup() override;

  // Setup unit conversion factors. The required units in the ROM are:
  // Cell dislocation density: m^-2
  // Wall dislocation density: m^-2 (if applicable)
  // irradiation dose rate: dpa/s. (if applicable)
  // stress: MPa
  // strain: nondim.
  // temperature: K
  virtual void setupUnitConversionFactors(const InputParameters & parameters);

  virtual void initQpStatefulProperties() override;
  virtual void
  computeStressInitialize(const GenericReal<is_ad> & effective_trial_stress,
                          const GenericRankFourTensor<is_ad> & elasticity_tensor) override;
  virtual GenericReal<is_ad> computeResidual(const GenericReal<is_ad> & effective_trial_stress,
                                             const GenericReal<is_ad> & scalar) override;

  virtual GenericReal<is_ad>
  computeDerivative(const GenericReal<is_ad> & /*effective_trial_stress*/,
                    const GenericReal<is_ad> & /*scalar*/) override
  {
    return _derivative;
  }

  virtual void
  computeStressFinalize(const GenericRankTwoTensor<is_ad> & plastic_strain_increment) override;
  virtual GenericReal<is_ad>
  maximumPermissibleValue(const GenericReal<is_ad> & effective_trial_stress) const override;
  virtual Real computeTimeStepLimit() override;

  void outputIterationSummary(std::stringstream * iter_output,
                              const unsigned int total_it) override;

  virtual void outputIterationStep(std::stringstream * iter_output,
                                   const GenericReal<is_ad> & effective_trial_stress,
                                   const GenericReal<is_ad> & scalar,
                                   const Real reference_residual) override;

  /// Enum to error, warn, ignore, or extrapolate if input is outside of window of applicability
  enum class WindowFailure
  {
    ERROR,
    WARN,
    IGNORE,
    EXCEPTION,
    DONOTHING,
    USELIMIT,
    EXTRAPOLATE
  };

  /**
   * Computes the ROM calculated increment for a given output and tile using custom input values.
   * @param out_index   Output index
   * @param input_values Custom input values to use instead of _input_values
   * @param derivative   Flag to return derivative of ROM increment with respect to stress.
   * @return  std::pair where:
   *          - first  = scalar ROM increment (y)
   *          - second = gradient vector doutput (size = number of spatial dimensions)
   */
  std::pair<GenericReal<is_ad>, std::vector<GenericReal<is_ad>>>
  computeROM(const unsigned out_index,
            std::vector<GenericReal<is_ad>> & input_values,
            const bool derivative);

  /**
   * Safely solves the linear system T * sol = rhs using LU decomposition.
   * Wraps the original DenseMatrix::lu_solve call with exception handling.
   * @tparam MatrixType Type of the matrix (e.g., DenseMatrix<Real>)
   * @tparam VecType Type of the vector (e.g., DenseVector<Real>)
   * @param T Matrix to decompose and solve
   * @param rhs Right-hand side vector
   * @param sol Solution vector (output)
   * @return true if the solve succeeded, false if an exception was caught
   */
  template <typename MatrixType, typename VecType>
  bool safe_lu_solve(MatrixType & T, const VecType & rhs, VecType & sol)
  {
    try
    {
      T.lu_solve(rhs, sol); // call the original DenseMatrix method
      return true;          // success
    }
    catch (const std::exception & e)
    {
      mooseWarning("LU solve exception: ", e.what());
      return false; // indicate failure
    }
    catch (...)
    {
      mooseWarning("LU solve unknown exception.");
      return false;
    }
  }

  /**
   * Prints the checks related to checkInBounds
   * @param inputString Description of where the check is being performed
   * @param input_values Optional vector of input values to check instead of _input_values
   */
  void printInBoundCheck(const std::string & inputString,
                         std::vector<GenericReal<is_ad>> & input_values);

  /**
   * Checks if the input combination is in a specific tile
   * @param input Input point
   * @param bound_min Minimum bound
   * @param bound_max Maximum bound
   * @return bool if in tile
   */
  bool checkInBounds(const GenericReal<is_ad> & input,
                     const GenericReal<is_ad> bound_min,
                     const GenericReal<is_ad> bound_max);

  /**
   * Convert the input variables into the form expected by the ROM Legendre polynomials to have a
   * normalized space from [-1, 1] so that every variable has equal weight
   * @param input Input value
   * @param node_min minimum node value of element
   * @param node_max maximum node value of element
   * @return Converted input
   */
  GenericReal<is_ad> isoparametricInput(const GenericReal<is_ad> & input,
                                        const GenericReal<is_ad> node_min,
                                        const GenericReal<is_ad> node_max);

  /**
   * Assemble the array of Legendre polynomials to be multiplied by the ROM coefficients
   * @param iso_input isoparamtric coordinate
   * @param derivative Optional flag to return derivative of converted input with respect to stress.
   */
  std::vector<GenericReal<is_ad>> iso_1d_lagrange_linear(const GenericReal<is_ad> iso_input,
                                                         const bool derivative = false);

  /**
   * Compute the sign of a value
   */
  template <typename T>
  int sgn(T val)
  {
    return (T(0) < val) - (val < T(0));
  }

  // Transform structures
  struct Log10Transform
  {
    Real factor;
    Real b;
    Real a;
    Real logmin;
    Real logmax;
  };

  struct CompressTransform
  {
    Real factor;
    Real compressor;
    Real original_min;
  };

  struct MinMaxTransform
  {
    Real data_min;
    Real data_max;
    Real scaled_min;
    Real scaled_max;
  };

  struct SymLogTransform
  {
    Real zmin;
    Real zmax;
    Real zbar;
    Real lowerbound;
    Real upperbound;

    SymLogTransform(Real zmin_in = 0.0,
                    Real zmax_in = 0.0,
                    Real zbar_in = 0.0,
                    Real lb = -1.0,
                    Real ub = 1.0)
      : zmin(zmin_in), zmax(zmax_in), zbar(zbar_in), lowerbound(lb), upperbound(ub)
    {
    }
  };

  using Transform =
      std::variant<Log10Transform, CompressTransform, MinMaxTransform, SymLogTransform>;

  template <typename T>
  void convertOutput(T & x,
                     const std::vector<std::pair<std::string, Transform>> & value_scalers,
                     const unsigned index,
                     const Real drom_output = 1.0,
                     const Real dt = 1.0,
                     const bool derivative = false)
  {
    // Ensure the index is within the valid range
    if (index >= value_scalers.size())
    {
      mooseWarning("Unknown index to transform: ", index);
      return;
    }

    // Get the transform pair (string key and Transform object)
    const auto & transform_pair = value_scalers[index];
    const auto & type = transform_pair.first;
    const auto & transform = transform_pair.second;

    if (std::holds_alternative<Log10Transform>(transform))
    {
      const auto & log_transform = std::get<Log10Transform>(transform);
      Real factor = log_transform.factor;
      Real b = log_transform.b;
      Real a = log_transform.a;
      Real logmin = log_transform.logmin;
      Real logmax = log_transform.logmax;

      // Perform transformations for Log10Transform
      if (type == "LOG10BOUNDED")
      {
        x = a * (std::log10(MetaPhysicL::raw_value(x + factor)) - logmin) / (logmax - logmin) + b;
        x *= dt;
      }

      else if (type == "EXP10BOUNDED")
      {

        if (derivative)
        {
          const auto u =
              std::pow(10, MetaPhysicL::raw_value((logmax - logmin) * (x - b) / a + logmin));
          x = u * std::log(10) * (logmax - logmin) / a * drom_output;
          x *= dt;
        }
        else
        {
          if (std::abs(MetaPhysicL::raw_value(x) - 0.0) < 1e-8)
            x = 0.0;
          else
          {
            const auto transformed =
                std::pow(10, MetaPhysicL::raw_value((x - b) * (logmax - logmin) / a + logmin));
            x = transformed - factor;
          }
          x *= dt;
        }
      }

      else if (type == "EXP10")
      {
        if (derivative)
        {
          x = std::pow(10, MetaPhysicL::raw_value(x)) * std::log(10) * drom_output;
          x *= dt;
        }
        else
        {
          x = std::pow(10, MetaPhysicL::raw_value(x));
          x *= dt;
        }
      }
    }

    else if (std::holds_alternative<CompressTransform>(transform))
    {
      const auto & compress_transform = std::get<CompressTransform>(transform);
      Real factor = compress_transform.factor;
      Real compressor = compress_transform.compressor;
      Real original_min = compress_transform.original_min;

      // Perform transformations
      if (type == "COMPRESS")
      {
        // Compress and scale the data
        auto sign_x = (x > 0) ? 1.0 : -1.0;
        if (x == 0)
          sign_x = 0.0; // Handle zero case if needed
        x = sign_x * std::pow(std::abs(MetaPhysicL::raw_value(x * factor)), compressor);
        x = std::log10(MetaPhysicL::raw_value(1 + x - original_min));
        x *= dt;
      }

      // Perform inverse transformation
      else if (type == "DECOMPRESS")
      {
        x = std::pow(10, MetaPhysicL::raw_value(x));
        x = x - 1 + original_min;
        x = sgn(x) * std::pow(std::abs(MetaPhysicL::raw_value(x)), (1 / compressor));
        x = x / factor;
        x *= dt;
      }
    }
    else if (std::holds_alternative<MinMaxTransform>(transform))
    {
      const auto & minmax_transform = std::get<MinMaxTransform>(transform);
      Real data_min = minmax_transform.data_min;
      Real data_max = minmax_transform.data_max;
      Real scaled_min = minmax_transform.scaled_min;
      Real scaled_max = minmax_transform.scaled_max;

      // Transformations
      if (type == "MINMAX")
      {
        if (derivative)
        {
          // Derivative of MinMax: constant scaling factor
          x = ((scaled_max - scaled_min) / (data_max - data_min)) * drom_output;
          x *= dt;
        }
        else
        {
          x = ((x - data_min) / (data_max - data_min)) * (scaled_max - scaled_min) + scaled_min;
          x *= dt;
        }
      }
      // Inverse transformations
      else if (type == "INVMINMAX")
      {
        if (derivative)
          {
            // Derivative of inverse MinMax: constant scaling factor
            x = ((data_max - data_min) / (scaled_max - scaled_min)) * drom_output;
            x *= dt;
          }
        else
        {
          x = ((x - scaled_min) / (scaled_max - scaled_min)) * (data_max - data_min) + data_min;
          x *= dt;
        }
      }
    }
    else if (std::holds_alternative<SymLogTransform>(transform))
    {
      const auto & sym_transform = std::get<SymLogTransform>(transform);
      Real lb = sym_transform.lowerbound;
      Real ub = sym_transform.upperbound;
      Real zmin = sym_transform.zmin;
      Real zmax = sym_transform.zmax;
      Real zbar = sym_transform.zbar;

      if (type == "SYMLOGBOUNDED")
      {
        // Forward transform: np.sign(x)*log10(1+abs(x)), scaled into [lb, ub]
        const auto Z = (x >= 0.0 ? std::log10(MetaPhysicL::raw_value(1.0 + x))
                                 : -std::log10(MetaPhysicL::raw_value(1.0 - x)));
        x = (ub - lb) * (Z - zbar) / (zmax - zmin);
        x *= dt;
      }

      else if (type == "EXPSYMLOGBOUNDED")
      {
        if (derivative)
        {
          // Derivative of inverse transform
          const auto X = (zmax - zmin) * x / (ub - lb) + zbar;
          const auto U = std::pow(10.0, std::abs(MetaPhysicL::raw_value(X)));
          x = U * std::log(10.0) * (zmax - zmin) / (ub - lb) * drom_output;
          x *= dt;
        }
        else
        {
          // Inverse transform
          const auto X = (zmax - zmin) * x / (ub - lb) + zbar;
          x = (X >= 0.0 ? (std::pow(10.0, MetaPhysicL::raw_value(X)) - 1.0)
                        : -(std::pow(10.0, MetaPhysicL::raw_value(-X)) - 1.0));
          x *= dt;
        }
      }
    }
  }

  /*
   * Returns vector of the functions to use for the conversion of input variables.
   * @return vector of the functions to use for the conversion of input variables.
   */
  virtual std::vector<std::pair<std::string, Transform>> getInputTransform()
  {
    return {}; // Default
  };

  /*
   * Returns vector of the functions to use for the conversion of output variables.
   * @return vector of the functions to use for the conversion of input variables.
   */
  virtual std::vector<std::pair<std::string, Transform>> getOutputTransform()
  {
    return {}; // Default
  };

  virtual std::vector<std::vector<GenericReal<is_ad>>> getInputLimits()
  {
    checkJSONKey("input_limits");
    
    auto limits_real = _json["input_limits"].template get<std::vector<std::vector<Real>>>();
    
    std::vector<std::vector<GenericReal<is_ad>>> result;
    result.reserve(limits_real.size());
    
    for (const auto& inner_vec : limits_real)
      result.emplace_back(inner_vec.begin(), inner_vec.end());
    
    return result;
  } 

  /*
   * Material specific node coordinates
   * variables
   * @return Node coordinates
   */
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodes()
  {
    checkJSONKey("nodes");
    
    auto nodes_real = _json["nodes"].template get<std::vector<std::vector<Real>>>();
    
    std::vector<std::vector<GenericReal<is_ad>>> result;
    result.reserve(nodes_real.size());
    
    for (const auto& inner_vec : nodes_real)
      result.emplace_back(inner_vec.begin(), inner_vec.end());
    
    return result;
  }

  /*
   * Material specific connectivity matrix
   * variables
   * @return Connectivity matrix
   */
  virtual std::vector<std::vector<unsigned int>> getConnectivityMatrix()
  {
    checkJSONKey("connectivity_matrix");
    return _json["connectivity_matrix"].template get<std::vector<std::vector<unsigned int>>>();
  }

  /*
   * Material specific, calibrated nodal values
   * variables
   * @return Calibrated nodal values
   */
  virtual std::vector<std::vector<GenericReal<is_ad>>> getNodalValues()
  {
    checkJSONKey("nodal_values");
    
    auto nodal_values_real = _json["nodal_values"].template get<std::vector<std::vector<Real>>>();
    
    std::vector<std::vector<GenericReal<is_ad>>> result;
    result.reserve(nodal_values_real.size());
    
    for (const auto& inner_vec : nodal_values_real)
      result.emplace_back(inner_vec.begin(), inner_vec.end());
    
    return result;
  }

  /*
   * Material specific, element lower bounds
   * variables
   * @return The element lower bounds (min of nodal values per element).
   */
  virtual std::vector<Eigen::MatrixXd> getMinValues() const
  {
    mooseError("No minValues defined in material class needed for computation. Abort.");
    return {};
  }

  /*
   * Material specific, element upper bounds
   * variables
   * @return The element lower bounds (min of nodal values per element).
   */
  virtual std::vector<Eigen::MatrixXd> getMaxValues() const
  {
    mooseError("No maxValues defined in material class needed for computation. Abort.");
    return {};
  }
  /*
   * Material specific, extrapolation element vector container
   * variables
   * @return Element numbers that are used for padding ("extrapolating outside of") the calibration
   * regime to handle occasional out-of-bounds situations
   */
  virtual std::vector<unsigned int> getExtrapolationElement()
  {
    checkJSONKey("extrapolation_element");
    return _json["extrapolation_element"].template get<std::vector<unsigned int>>();
  }

  /*
   * Material specific orientations of variables.
   * @return Vector of numbers of elements
   */
  virtual std::vector<unsigned int> getElementNumbers()
  {
    if (_json.contains("element_numbers"))
      return _json["element_numbers"].template get<std::vector<unsigned int>>();

    // Default to 1 element per dimension
    std::vector<unsigned int> element_numbers(_num_inputs, 1);
    return element_numbers;
  };

  /*
   * Material specific triangular dimensions.
   * @return Vector of dimension indices that should be treated as triangular (simplex)
   */
  virtual std::vector<unsigned int> getTriangularDimensions()
  {
    // Default implementation: first two dimensions are triangular if there are at least 2
    // dimensions
    if (_num_inputs >= 2)
      return {0, 1};
    return {};
  };

  /*
   * Degree of shape functions
   *
   * @return string declaring the shape function degree
   */
  virtual std::string getShapeFunctionDegree() { return "linear"; }

  /// Coupled temperature variable
  const GenericVariableValue<is_ad> & _temperature;

  /// Optionally coupled environmental factor
  const GenericMaterialProperty<Real, is_ad> * _environmental;

  /// Optionally coupled microstructure paremeters
  const GenericMaterialProperty<Real, is_ad> * _microstruct_param_1;
  const GenericMaterialProperty<Real, is_ad> * _microstruct_param_2;
  const GenericMaterialProperty<Real, is_ad> * _microstruct_param_3;

  /*
   * Vector of vectors WindowFailure enum that informs how to handle input that is outside of the
   * limits. Shape is number of inputs by 2 (lower and upper window enum)
   */
  std::vector<std::pair<WindowFailure, WindowFailure>> _window_failure;

  /// Flag to output verbose infromation
  const bool _verbose;

 

  /// (NEW) Whether to use dislocations based on parameter presence
  const bool _use_cell_dislocations;

  ///@{Material properties for cell (glissile) dislocation densities (1/m^2)
  GenericMaterialProperty<Real, is_ad> * _cell_dislocations;
  const MaterialProperty<Real> * _cell_dislocations_old;
  ///@}

  /// Maximum cell dislocation increment
  Real _max_cell_increment;

  /// Optional cell dislocation forcing function
  const Function * const _cell_function;

  /// Container for cell dislocation increment
  GenericReal<is_ad> _cell_dislocation_increment;

  /// Flag to indicate if substepping was applied in computeStressFinalize
  bool _substepping_applied;


  /// (NEW) Whether to use dislocations based on parameter presence
  const bool _use_wall_dislocations;
  ///@{Material properties for wall (immobile) dislocation densities (1/m^2)
  GenericMaterialProperty<Real, is_ad> * _wall_dislocations;
  const MaterialProperty<Real> * _wall_dislocations_old;
  ///@}

  /// Maximum wall dislocation increment
  Real _max_wall_increment;

  /// Optional wall dislocation forcing function
  // const Function * const _wall_function; // Old when working with -dbg
  const Function * const _wall_function;

  /// Container for wall dislocation increment
  GenericReal<is_ad> _wall_dislocation_increment;

  /// Optiontal effective stress forcing function
  const Function * const _stress_function;

  /// Input indices for various parameters
  std::vector<unsigned int> _input_indices;

  /// Output indices for various parameters
  std::vector<unsigned int> _output_indices;

  /// Map of input keys to indices
  std::map<std::string, unsigned int> _input_key_to_index;

  /// Map of output keys to indices
  std::map<std::string, unsigned int> _output_key_to_index;

  /// Specific input indices for common parameters
  unsigned int _cell_input_index;
  unsigned int _wall_input_index;
  unsigned int _stress_input_index;
  unsigned int _old_strain_input_index;
  unsigned int _temperature_input_index;
  unsigned int _environmental_input_index;
  unsigned int _microstruct_param_1_input_index;
  unsigned int _microstruct_param_2_input_index;
  unsigned int _microstruct_param_3_input_index;

  /// Specific output indices for common parameters
  unsigned int _cell_output_index;
  unsigned int _wall_output_index;
  unsigned int _strain_output_index;

  /**
   * Initialize the key-to-index maps with default values
   */
  virtual void initializeKeyMaps();

  /**
   * Get input index by key
   * @param key The key for the input parameter
   * @return The index for the input parameter
   */
  unsigned int getInputIndexByKey(const std::string & key) const;

  /**
   * Get output index by key
   * @param key The key for the output parameter
   * @return The index for the output parameter
   */
  unsigned int getOutputIndexByKey(const std::string & key) const;

  /**
   * Set input index by key
   * @param key The key for the input parameter
   * @param index The index for the input parameter
   */
  void setInputIndexByKey(const std::string & key, unsigned int index);

  /**
   * Set output index by key
   * @param key The key for the output parameter
   * @param index The index for the output parameter
   */
  void setOutputIndexByKey(const std::string & key, unsigned int index);

  /// Optional old creep strain forcing function
  const Function * const _creep_strain_old_forcing_function;

  /// Number of elements
  unsigned int _num_elements;

  /// Number of inputs for the ROM data set
  unsigned int _num_inputs;

  /// Number of inputs to the ROM data set
  unsigned int _num_outputs;

  /// Total number of Legendre polynomial coefficients for the ROM data set in each parition
  std::vector<unsigned int> _num_coefs;

  /// Output transform rules defined by the ROM data set for each input
  std::vector<std::pair<std::string, Transform>> _input_transform;

  /// Output transform rules defined by the ROM data set for each output
  std::vector<std::pair<std::string, Transform>> _output_transform;

  /// Input limits defined by the ROM data set
  std::vector<std::vector<GenericReal<is_ad>>>  _input_limits;

  /// Nodal coordinates of the finite element surrogate model
  std::vector<std::vector<GenericReal<is_ad>>> _nodes;

  /// Connectivity matrix for the finite element surrogate model
  std::vector<std::vector<unsigned int>> _connectivity_matrix;

  /// Calibrated nodal values of the finite element surrogate model
  std::vector<std::vector<GenericReal<is_ad>>>  _nodal_values;

  // /// Calibrated nodal values of the finite element surrogate model
  std::vector<unsigned int> _extrapolation_element;

  /// minimum nodal values per element of the finite element surrogate model
  std::vector<Eigen::MatrixXd> _min_nodal_values;

  /// maximum nodal values per element of the finite element surrogate model
  std::vector<Eigen::MatrixXd> _max_nodal_values;

  /// Creep rate material property
  GenericMaterialProperty<Real, is_ad> & _creep_rate;

  /// Cell dislocations rate of change
  GenericMaterialProperty<Real, is_ad> * _cell_rate;

  /// Wall dislocations rate of change
  GenericMaterialProperty<Real, is_ad> * _wall_rate;

  /// Container for derivative of creep increment with respect to strain
  GenericReal<is_ad> _derivative;

  /// Container for input values
  std::vector<GenericReal<is_ad>> _input_values;

  /// Container for old input values
  std::vector<Real> _old_input_values;

  /// Container for global limits
  std::vector<std::pair<Real, Real>> _global_limits;

  /// Unit conversion factors required to convert from the specified unit to MPa
  Real _stress_ucf;

  ///@{Material properties accumulated at substeps
  GenericMaterialProperty<Real, is_ad> * _wall_dislocations_step;
  GenericMaterialProperty<Real, is_ad> * _cell_dislocations_step;
  ///@}

  /// Total plastic strain increment in step (summing substep contributions)
  RankTwoTensor _plastic_strain_increment;

  /// Material property capturing number of substeps for output purposes (defaults to one if substepping isn't used)
  MaterialProperty<Real> & _number_of_substeps;

  /// Timers to expose via postprocessors
  GenericMaterialProperty<Real, is_ad> & _duration_element_search;

  /// Active element in ROM that contains the input point for the query
  MaterialProperty<Real> & _active_element;

  /// Expose brute force fallback
  GenericMaterialProperty<Real, is_ad> & _fallback_brute_force;

  /** Build the tree from the ROM mesh (called from `initialSetup`). */
  void initializeRtreeIndex();

  /** Query the tree for the element that contains `point`. Returns -1 if not found. */
  int findElementUsingRtree(const std::vector<GenericReal<is_ad>> & point) const;

  KDTreeAdaptor * _kdtree{nullptr};
  std::vector<std::vector<GenericReal<is_ad>>> _master_points;
  std::vector<std::vector<Real>> _master_points_real;
  bool _use_kdtree_indexing{true};

  std::unique_ptr<ROMAdaptor> _adaptor; // or ROMAdaptor* _adaptor;

  /// check if a JSON file was loaded and if the specified key exists
  void checkJSONKey(const std::string & key);

  /// JSON object constructed from the datafile
  nlohmann::json _json;

  using Material::_dt;
  using Material::_name;
  using Material::_q_point;
  using Material::_qp;
  using Material::_t;
  using Material::coupledGenericValue;
  using RadialReturnCreepStressUpdateBaseTempl<is_ad>::computeResidual;
  using RadialReturnCreepStressUpdateBaseTempl<is_ad>::computeDerivative;
  using RadialReturnCreepStressUpdateBaseTempl<is_ad>::_apply_strain;
  using RadialReturnCreepStressUpdateBaseTempl<is_ad>::initQpStatefulProperties;
  using RadialReturnCreepStressUpdateBaseTempl<is_ad>::outputIterationStep;
  using RadialReturnCreepStressUpdateBaseTempl<is_ad>::outputIterationSummary;
};

typedef LAROManceStressUpdateBaseUniversalTempl<false> LAROManceStressUpdateBaseUniversal;
typedef LAROManceStressUpdateBaseUniversalTempl<true> ADLAROManceStressUpdateBaseUniversal;
