/* ========================================================================= *
 *                                                                           *
 *                                 STAR++                                    *
 *                    Copyright (c) 2018, Chongyi Zheng.                     *
 *                          All Rights reserved.                             *
 *                                                                           *
 * ------------------------------------------------------------------------- *
 *                                                                           *
 * This software implements a 3D human skinning model - STAR: A Skinned      *
 * Multi-Person Linear Model with C++.                                       *
 *                                                                           *
 * For more detail, see the paper published by Max Planck Institute for      *
 * Intelligent Systems on SIGGRAPH ASIA 2015.                                *
 *                                                                           *
 * We provide this software for research purposes only.                      *
 * The original STAR model is available at http://smpl.is.tue.mpg.           *
 *                                                                           *
 * ========================================================================= */

//=============================================================================
//
//  CLASS STAR IMPLEMENTATIONS
//
//=============================================================================


//===== EXTERNAL MACROS =======================================================


//===== INCLUDES ==============================================================

//----------
#include <fstream>
#include <experimental/filesystem>
//----------
#include "cnpy.h"
//----------
#include "definition/def.h"
#include "toolbox/TorchEx.hpp"
#include "smpl/STAR.h"
//----------

//===== EXTERNAL FORWARD DECLARATIONS =========================================


//===== NAMESPACES ============================================================

namespace smpl {

//===== INTERNAL MACROS =======================================================


//===== INTERNAL FORWARD DECLARATIONS =========================================


//===== CLASS IMPLEMENTATIONS =================================================

/**STAR
 * 
 * Brief
 * ----------
 * 
 *      Default constructor.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
STAR::STAR() noexcept(true) :
    m__device(torch::kCPU),
    m__modelPath(),
    m__vertPath(),
    m__faceIndices(),
    m__shapeBlendBasis(),
    m__poseBlendBasis(),
    m__templateRestShape(),
    m__jointRegressor(),
    m__kinematicTree(),
    m__weights(),
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{
}

/**STAR (overload)
 * 
 * Brief
 * ----------
 * 
 *      Constructor to initialize model path.
 * 
 * Arguments
 * ----------
 * 
 *      @modelPath: - string -
 *          Model path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
STAR::STAR(std::string &modelPath,
    std::string &vertPath, torch::Device &device) noexcept(false) :
    m__device(torch::kCPU),
    m__blender(),
    m__regressor(),
    m__transformer(),
    m__skinner()
{
    if (device.has_index()) {
        m__device = device;
    }
    else {
        throw smpl_error("STAR", "Failed to fetch device index!");
    }

    std::experimental::filesystem::path path(modelPath);
    if (std::experimental::filesystem::exists(path)) {
        m__modelPath = modelPath;
        m__vertPath = vertPath;
    }
    else {
        throw smpl_error("STAR", "Failed to initialize model path!");
    }
}

/**STAR (overload)
 * 
 * Brief
 * ----------
 * 
 *      Copy constructor.
 * 
 * Arguments
 * ----------
 * 
 *      @smpl: - const STAR& -
 *          The <LinearBlendSkinning> instantiation to copy with.
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
STAR::STAR(const STAR& smpl) noexcept(false) :
    m__device(torch::kCPU)
{
    try {
        *this = smpl;
    }
    catch(std::exception &e) {
        throw;
    }
}

/**~STAR
 * 
 * Brief
 * ----------
 * 
 *      Destructor
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
STAR::~STAR() noexcept(true)
{
}

/**operator=
 * 
 * Brief
 * ----------
 * 
 *      Assignment is used to copy a <STAR> instantiation.
 * 
 * Arguments
 * ----------
 * 
 *      @smpl: - const STAR& -
 *          The <STAR> instantiation to copy with.
 * 
 * Return
 * ----------
 * 
 *      @this*: - STAR & -
 *          Current instantiation.
 * 
 */
STAR &STAR::operator=(const STAR& smpl) noexcept(false)
{
    //
    // hard copy
    //
    if (smpl.m__device.has_index()) {
        m__device = smpl.m__device;
    }
    else {
        throw smpl_error("STAR", "Failed to fetch device index!");
    }

    std::experimental::filesystem::path path(smpl.m__modelPath);
    if (std::experimental::filesystem::exists(path)) {
        m__modelPath = smpl.m__modelPath;
    }
    else {
        throw smpl_error("STAR", "Failed to copy model path!");
    }

    try {
        m__vertPath = smpl.m__vertPath;

        m__blender = smpl.m__blender;
        m__regressor = smpl.m__regressor;
        m__transformer = smpl.m__transformer;
        m__skinner = smpl.m__skinner;
    }
    catch(std::exception &e) {
        throw;
    }

    //
    // soft copy
    //
    if (smpl.m__faceIndices.sizes() ==
        torch::IntArrayRef({FACE_INDEX_NUM, 3})) {
        m__faceIndices = smpl.m__faceIndices.clone().to(m__device);
    }

    if (smpl.m__shapeBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, SHAPE_BASIS_DIM})) {
        m__shapeBlendBasis = smpl.m__shapeBlendBasis.clone().to(
            m__device);
    }

    if (smpl.m__poseBlendBasis.sizes() == 
        torch::IntArrayRef({VERTEX_NUM, 3, SMPL_POSE_BASIS_DIM})) {
        m__poseBlendBasis = smpl.m__poseBlendBasis.clone().to(m__device);
    }

    if (smpl.m__jointRegressor.sizes() == 
        torch::IntArrayRef({JOINT_NUM, VERTEX_NUM})) {
        m__jointRegressor = smpl.m__jointRegressor.clone().to(m__device);
    }

    if (smpl.m__templateRestShape.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, 3})) {
        m__templateRestShape = smpl.m__templateRestShape.clone().to(
            m__device);
    }

    if (smpl.m__kinematicTree.sizes() ==
        torch::IntArrayRef({2, JOINT_NUM})) {
        m__kinematicTree = smpl.m__kinematicTree.clone().to(m__device);
    }

    if (smpl.m__weights.sizes() ==
        torch::IntArrayRef({VERTEX_NUM, JOINT_NUM})) {
        m__weights = smpl.m__weights.clone().to(m__device);
    }

    return *this;
}

/**setDevice
 * 
 * Brief
 * ----------
 * 
 *      Set the torch device.
 * 
 * Arguments
 * ----------
 * 
 *      @device: - const Device & -
 *          The torch device to be used.
 * 
 * Return
 * ----------
 * 
 */
void STAR::setDevice(const torch::Device &device) noexcept(false)
{
    if (device.has_index()) {
        m__device = device;
        m__blender.setDevice(device);
        m__regressor.setDevice(device);
        m__transformer.setDevice(device);
        m__skinner.setDevice(device);
    }
    else {
        throw smpl_error("STAR", "Failed to fetch device index!");
    }

    return;
}

/**setModelPath
 * 
 * Brief
 * ----------
 * 
 *      Set model path to the JSON model file.
 * 
 * Arguments
 * ----------
 * 
 *      @modelPath: - string -
 *          Model path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
void STAR::setModelPath(const std::string &modelPath) noexcept(false)
{
    std::experimental::filesystem::path path(modelPath);
    if (std::experimental::filesystem::exists(path)) {
        m__modelPath = modelPath;
    }
    else {
        throw smpl_error("STAR", "Failed to initialize model path!");
    }

    return;
}

/**setVertPath
 * 
 * Brief
 * ----------
 * 
 *      Set path for exporting the mesh to OBJ file.
 * 
 * Arguments
 * ----------
 * 
 *      @vertexPath: - string -
 *          Vertex path to be specified.
 * 
 * Return
 * ----------
 * 
 * 
 */
void STAR::setVertPath(const std::string &vertexPath) noexcept(false)
{
    m__vertPath = vertexPath;

    return;
}

/**getRestShape
 * 
 * Brief
 * ----------
 * 
 *      Get deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @restShape: - Tensor -
 *          Deformed shape in rest pose, (N, 6890, 3)
 * 
 */
torch::Tensor STAR::getRestShape() noexcept(false)
{
    torch::Tensor restShape;
    
    try {
        restShape = m__regressor.getRestShape().clone().to(m__device);
    }
    catch(std::exception &e) {
        throw;
    }

    return restShape;
}

/**getFaceIndex
 * 
 * Brief
 * ----------
 * 
 *      Get vertex indices of each face.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @faceIndices: - Tensor -
 *          Vertex indices of each face (triangles), (13776, 3).
 * 
 */
torch::Tensor STAR::getFaceIndex() noexcept(false)
{
    torch::Tensor faceIndices;
    if (m__faceIndices.sizes() !=
        torch::IntArrayRef(
            {FACE_INDEX_NUM, 3})) {
        faceIndices = m__faceIndices.clone().to(m__device);
    }
    else {
        throw smpl_error("STAR", "Failed to get face indices!");
    }

    return faceIndices;
}

/**getRestJoint
 * 
 * Brief
 * ----------
 * 
 *      Get joint locations of the deformed shape in rest pose.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @joints: - Tensor -
 *          Joint locations of the deformed mesh in rest pose, (N, 24, 3).
 * 
 */
torch::Tensor STAR::getRestJoint() noexcept(false)
{
    torch::Tensor joints;
    
    try {
        joints = m__regressor.getJoint().clone().to(m__device);
    }
    catch (std::exception &e) {
        throw;
    }

    return joints;
}

/**getVertex
 * 
 * Brief
 * ----------
 * 
 *      Get vertex locations of the deformed mesh.
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 *      @vertices: - Tensor -
 *          Vertex locations of the deformed mesh, (N, 6890, 3).
 * 
 */
torch::Tensor STAR::getVertex() noexcept(false)
{
    torch::Tensor vertices;

    try {
        vertices = m__skinner.getVertex().clone().to(m__device);
    }
    catch(std::exception &e) {
        throw;
    }

    return vertices;
}

/**init
 * 
 * Brief
 * ----------
 * 
 *          Load model data stored as JSON file into current application.
 *          (Note: The loading will spend a long time because of a large
 *           JSON file.)
 * 
 * Arguments
 * ----------
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void STAR::init() noexcept(false)
{
    std::experimental::filesystem::path path(m__modelPath);
    if (std::experimental::filesystem::exists(path)) {
        auto data = cnpy::npz_load(path);

        //
        // data loading
        //
        // face indices
        auto& faceIndices = data["f"];
        assert(faceIndices.shape.size() == 2);
        assert(faceIndices.shape[0] == FACE_INDEX_NUM);
        assert(faceIndices.shape[1] == 3);
        m__faceIndices = torch::from_blob(faceIndices.data<int32_t>(),
                                          {FACE_INDEX_NUM, 3}, torch::kInt32).clone().to(
                m__device);

        // blender
        auto& shapeBlendShapes = data["shapedirs"];
        assert(shapeBlendShapes.shape.size() == 3);
        assert(shapeBlendShapes.shape[0] == VERTEX_NUM);
        assert(shapeBlendShapes.shape[1] == 3);
        assert(shapeBlendShapes.shape[2] == SHAPE_BASIS_DIM);
        auto& poseBlendShapes = data["posedirs"];
        assert(poseBlendShapes.shape.size() == 3);
        assert(poseBlendShapes.shape[0] == VERTEX_NUM);
        assert(poseBlendShapes.shape[1] == 3);
        assert(poseBlendShapes.shape[2] == SMPL_POSE_BASIS_DIM);
        m__shapeBlendBasis = torch::from_blob(shapeBlendShapes.data<float>(),
                                              {VERTEX_NUM, 3, SHAPE_BASIS_DIM}).to(m__device);// (6890, 3, 10)
        m__poseBlendBasis = torch::from_blob(poseBlendShapes.data<float>(),
                                             {VERTEX_NUM, 3, SMPL_POSE_BASIS_DIM}).to(m__device);// (6890, 3, 93)

        // regressor
        auto& templateRestShape = data["v_template"];
        assert(templateRestShape.shape.size() == 2);
        assert(templateRestShape.shape[0] == VERTEX_NUM);
        assert(templateRestShape.shape[1] == 3);
        auto& jointRegressor = data["J_regressor"];
        assert(jointRegressor.shape.size() == 2);
        assert(jointRegressor.shape[0] == JOINT_NUM);
        assert(jointRegressor.shape[1] == VERTEX_NUM);
        m__templateRestShape = torch::from_blob(templateRestShape.data<float>(),
            {VERTEX_NUM, 3}).to(m__device);// (6890, 3)
        m__jointRegressor = torch::from_blob(jointRegressor.data<float>(),
            {JOINT_NUM, VERTEX_NUM}).to(m__device);// (24, 6890)

        // transformer
        auto& kinematicTree = data["kintree_table"];
        assert(kinematicTree.shape.size() == 2);
        assert(kinematicTree.shape[0] == 2);
        assert(kinematicTree.shape[1] == JOINT_NUM);
        m__kinematicTree = torch::from_blob(kinematicTree.data<int64_t>(),
            {2, JOINT_NUM}, torch::kInt64).to(m__device);// (2, 24)

        // skinner
        auto& weights = data["weights"];
        assert(weights.shape.size() == 2);
        assert(weights.shape[0] == VERTEX_NUM);
        assert(weights.shape[1] == JOINT_NUM);
        m__weights = torch::from_blob(weights.data<float>(),
            {VERTEX_NUM, JOINT_NUM}).to(m__device);// (6890, 24)
    }
    else {
        throw smpl_error("STAR", "Cannot initialize a STAR model!");
    }

    return;
}

/**launch
 * 
 * Brief
 * ----------
 * 
 *          Run the model with a specific group of beta, theta, and 
 *          translation.
 * 
 * Arguments
 * ----------
 * 
 *      @beta: - Tensor -
 *          Batch of shape coefficient vectors, (N, 10).
 * 
 *      @theta: - Tensor -
 *          Batch of pose in axis-angle representations, (N, 24, 3).
 * 
 *      @translation: - Tensor -
 *          Batch of global translation vectors, (N, 3).
 * 
 * 
 * Return
 * ----------
 * 
 * 
 */
void STAR::launch(
    torch::Tensor &beta, 
    torch::Tensor &theta) noexcept(false)
{
    if (beta.sizes() != torch::IntArrayRef({BATCH_SIZE, SHAPE_BASIS_DIM})
        && theta.sizes() != torch::IntArrayRef({BATCH_SIZE, JOINT_NUM, 3})) {

        throw smpl_error("STAR", "Cannot launch a STAR model!");
    }

    try {
        //
        // blend shapes
        //
        m__blender.setBeta(beta);
        m__blender.setTheta(theta);
        m__blender.setShapeBlendBasis(m__shapeBlendBasis);
        m__blender.setPoseBlendBasis(m__poseBlendBasis);

        m__blender.blend();

        torch::Tensor shapeBlendShape = m__blender.getShapeBlendShape();
        torch::Tensor poseBlendShape = m__blender.getPoseBlendShape();
        torch::Tensor poseRotation = m__blender.getPoseRotation();

        //
        // regress joints
        //
        m__regressor.setTemplateRestShape(m__templateRestShape);
        m__regressor.setJointRegressor(m__jointRegressor);
        m__regressor.setShapeBlendShape(shapeBlendShape);
        m__regressor.setPoseBlendShape(poseBlendShape);

        m__regressor.regress();

        torch::Tensor restShape = m__regressor.getRestShape();
        torch::Tensor joints = m__regressor.getJoint();

        //
        // transform
        //
        m__transformer.setKinematicTree(m__kinematicTree);
        m__transformer.setJoint(joints);
        m__transformer.setPoseRotation(poseRotation);

        m__transformer.transform();

        torch::Tensor transformation = m__transformer.getTransformation();

        //
        // skinning
        //
        m__skinner.setWeight(m__weights);
        m__skinner.setRestShape(restShape);
        m__skinner.setTransformation(transformation);

        m__skinner.skinning();
    }
    catch(std::exception &e) {
        throw;
    }

    return;
}

/**out
 * 
 * Brief
 * ----------
 * 
 *      Export the deformed mesh to OBJ file.
 * 
 * Arguments
 * ----------
 * 
 *      @index: - size_t -
 *          A mesh in the batch to be exported.
 * 
 * Return
 * ----------
 * 
 * 
 */
void STAR::out(int64_t index) noexcept(false)
{
    torch::Tensor vertices = 
        m__skinner.getVertex().clone().to(m__device);// (N, 6890, 3)

    if (vertices.sizes() ==
            torch::IntArrayRef(
                {BATCH_SIZE, VERTEX_NUM, 3})
        && m__faceIndices.sizes() ==
            torch::IntArrayRef(
                {FACE_INDEX_NUM, 3})
        ) {
        std::ofstream file(m__vertPath);

        torch::Tensor slice_ = TorchEx::indexing(vertices,
            torch::IntList({index}));// (6890, 3)

        auto slice = slice_.accessor<float, 2>();
        auto faceIndices = m__faceIndices.accessor<int32_t, 2>();

        for (int64_t i = 0; i < VERTEX_NUM; i++) {
            file << 'v' << ' '
                << slice[i][0] << ' '
                << slice[i][1] << ' '
                << slice[i][2] << '\n';
        }

        for (int64_t i = 0; i < FACE_INDEX_NUM; i++) {
            file << 'f' << ' '
                << faceIndices[i][0] << ' '
                << faceIndices[i][1] << ' '
                << faceIndices[i][2] << '\n';
        }
    }
    else {
        throw smpl_error("STAR", "Cannot export the deformed mesh!");
    }

    return;
}


//=============================================================================
} // namespace smpl
//=============================================================================
