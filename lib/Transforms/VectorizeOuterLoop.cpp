//===- LoopUnrollAndJam.cpp - Code to perform outerloop vectorization ---------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Transforms/Passes.h"
#include "stack"

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"

#include "mlir/IR/StandardTypes.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/NestedMatcher.h"
#include "mlir/Support/Functional.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"


using namespace mlir;


#define DEBUG_TYPE "affine-outerloop-vectorize"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");
namespace mlir{
  LogicalResult vectorizeOuterLoopByWidth(AffineForOp forOp,
                                          uint64_t vectorWidth);
}

namespace {
// Nested Pattern to detect loops that are already vectorized
// Required to provide this to isVectorizableLoopBody() function
static NestedPattern &vectorMemOpPattern() {
  static auto pattern = matcher::Op([](Operation &op) {
	if (auto load = dyn_cast<AffineLoadOp>(op)) {
	  auto memRefType = load.getMemRef()->getType().template cast<MemRefType>();
	  auto type = memRefType.getElementType();
	  if(type.isa<VectorType>())
	    return true;
	}
	else if (auto store = dyn_cast<AffineStoreOp>(op)) {
	  auto memRefType = store.getMemRef()->getType().template cast<MemRefType>();
      auto type = memRefType.getElementType();
      if(type.isa<VectorType>())
        return true;
    }
	return false;
  });
  return pattern;
}

struct VectorizeOuterLoop : public FunctionPass<VectorizeOuterLoop> {
  Optional<unsigned> vectorWidth;
  static const unsigned DefaultvectorWidth = 4;
  DenseMap<Value *, Value *> operandReplacement;
  std::stack<Operation*> operationsToDelete;
  std::stack<Operation*> vectorOperations;

  explicit VectorizeOuterLoop(Optional<unsigned> vectorWidth = None)
      : vectorWidth(vectorWidth) {}
  
  LogicalResult vectorizeNonLoadandStore(Operation *opInst, uint64_t vectorWidth);
  template <typename LoadOrStoreOpPointer>
  LogicalResult vectorizeLoadStore(LoadOrStoreOpPointer memoryOp, 
                                   AffineForOp forOp, uint64_t vectorWidth);
  LogicalResult vectorizeOuterLoopByWidth(AffineForOp forOp,
                                          uint64_t vectorWidth);

  void runOnFunction() override;

  LogicalResult runOnAffineForOp(AffineForOp forOp);
};
} // end anonymous namespace


void VectorizeOuterLoop::runOnFunction() {
  NestedPatternContext mlContext;
  FuncOp f = getFunction();
  std::vector<Operation *> parallelLoops;
  f.walk([&parallelLoops](AffineForOp loop) {
    if (isLoopParallel(loop)){
      int fastestMovingDim;
      bool vectorizable = isVectorizableLoopBody(loop, &fastestMovingDim, vectorMemOpPattern());
      if(vectorizable && fastestMovingDim == 0){
        parallelLoops.push_back(loop);
      }
    }
  });

  for(auto op:parallelLoops){
    auto forOp = dyn_cast<AffineForOp>(op);
    runOnAffineForOp(forOp);
  }
  
}


LogicalResult VectorizeOuterLoop::runOnAffineForOp(AffineForOp forOp) {
  // vectorize by the factor that was passed if any.
  if (vectorWidth.hasValue())
    return vectorizeOuterLoopByWidth(forOp, vectorWidth.getValue());
  // vectorize by four otherwise.
  return vectorizeOuterLoopByWidth(forOp, DefaultvectorWidth);
}


LogicalResult VectorizeOuterLoop::vectorizeOuterLoopByWidth(AffineForOp forOp,
                                          uint64_t vectorWidth) {
  using namespace functional;

  if(!isLoopParallel(forOp)){
    llvm::errs() << "Loop is not parallel\n";
    return failure();
  }
  

  int64_t step = forOp.getStep();
  forOp.setStep(step * vectorWidth);
   
  auto loadAndStores = matcher::Op();
  SmallVector<NestedMatch, 8> loadAndStoresMatches;
  loadAndStores.match(forOp.getOperation(), &loadAndStoresMatches);

  for (auto ls : loadAndStoresMatches) {
    auto *opInst = ls.getMatchedOperation();
    auto load = dyn_cast<AffineLoadOp>(opInst);
    auto store = dyn_cast<AffineStoreOp>(opInst);
    if(load){
      vectorizeLoadStore(load, forOp, vectorWidth);
    }else if(store){
      vectorizeLoadStore(store, forOp, vectorWidth);
    }else if(!isa<AffineTerminatorOp>(opInst)){
      vectorizeNonLoadandStore(opInst, vectorWidth);
    }
  }
  while(operationsToDelete.size()){
        auto opInst = operationsToDelete.top();
        opInst->erase();
        operationsToDelete.pop();
  }
  
  return success();
}

// Vectorize an operation in the loop which is not load or store
// It which check if the operand is already vectorized. If the 
// operand is defined in the loop, It would have been vectorized.
// If not, add a splat operation just after the operation that 
// defines the operand. Then create a new operation with vector 
// operands.
LogicalResult VectorizeOuterLoop::vectorizeNonLoadandStore(Operation *opInst, uint64_t vectorWidth){
  if (opInst->getNumRegions() != 0)
    return success();
  
  // Generate corresponding vector types for operands
  SmallVector<Type, DefaultvectorWidth> vectorTypes;
  for (auto *v : opInst->getResults()) {
    vectorTypes.push_back(
        VectorType::get(llvm::SmallVector<int64_t, 4>
                       (1, vectorWidth), v->getType()));
  }
  // Create vector operand. Used already vectorized stores,
  // if not vectorized, add splat operations
  SmallVector<Value *, 8> vectorOperands;
  for (auto *operand : opInst->getOperands()) {
    auto vectorOperand = operandReplacement.find(operand);
    if(vectorOperand != operandReplacement.end()){
      vectorOperands.push_back(vectorOperand->second);
    }
    else{
      auto operandType = operand->getType();
      auto vectorType = VectorType::get(llvm::SmallVector<int64_t, 4>
                                       (1, vectorWidth), operandType);
      OpBuilder b1(operand->getParentRegion());
      auto definingOp = operand->getDefiningOp();
      if(definingOp){
          b1.setInsertionPointAfter(definingOp);
      }
      auto newSplatOp = b1.create<mlir::SplatOp>(
          opInst->getLoc(), vectorType, operand);
      operandReplacement.insert(std::make_pair(operand, newSplatOp.getResult()));
      vectorOperands.push_back(newSplatOp.getResult());
      vectorOperations.push(newSplatOp);
    }
  }

  OpBuilder b(opInst);
  OperationState newOp(opInst->getLoc(), opInst->getName().getStringRef(),
                      vectorOperands, vectorTypes, opInst->getAttrs(),
                      /*successors=*/{},
                      /*regions=*/{}, opInst->hasResizableOperandsList());
  auto newOpInst = b.createOperation(newOp);
  operandReplacement.insert(std::make_pair(opInst->getResult(0), newOpInst->getResult(0)));
  operationsToDelete.push(opInst);
  vectorOperations.push(newOpInst);
  return success();
}

// Function to vectorize load and store operations. In case of a
// store, It will create a subview of the MemRef with vector 
// elements and create a store operations that stores the vector
// to the subview
// In case of load operations, it will check if the load is
// contiguous or invariant. If load is invariant, it will create
// a splat operation after load so that the loaded value is 
// now a vector. If load is varying acoress the iv of this loop, 
// it will create a subview with vector types and load a vector
// value from that.
template <typename LoadOrStoreOpPointer>
LogicalResult VectorizeOuterLoop::vectorizeLoadStore(LoadOrStoreOpPointer memoryOp, 
                                          AffineForOp forOp, uint64_t vectorWidth){
  auto iv = forOp.getInductionVar();
  auto memRefType = memoryOp.getMemRef()->getType().template cast<MemRefType>();
  auto elementType = memRefType.getElementType();
  auto vectorType = VectorType::get(llvm::SmallVector<int64_t, 4>
                                   (1, vectorWidth), elementType);

  auto originalShape = memRefType.getShape();
  std::vector <int64_t> targetShape(originalShape);

  targetShape[targetShape.size()-1] = targetShape[targetShape.size()-1] / DefaultvectorWidth;
  auto targetMemref = memRefType.get(targetShape, vectorType);
  auto *opInst = memoryOp.getOperation();
  
  OpBuilder b(opInst);
  // Creating a builder in the region where the MemRef is defined.
  // In cases like function arguments, the subview will be created
  // as the first operations of function
  OpBuilder subViewBuilder(memoryOp.getMemRef()->getParentRegion());
  // If there is a defining operations for the MemRef, change the
  // insertion point to after that defining operations
  auto definingOp = memoryOp.getMemRef()->getDefiningOp();
  if(definingOp){
    subViewBuilder.setInsertionPointAfter(definingOp);
  }
  SmallVector<Value *, 8> indices;
  indices.append(memoryOp.getMapOperands().begin(),
                 memoryOp.getMapOperands().end());
    
  if (auto store = dyn_cast<AffineStoreOp>(opInst)) {
    auto valueToStore = store.getValueToStore();
    auto vectorValue = operandReplacement.find(valueToStore);
    if(vectorValue == operandReplacement.end())
        return failure(); /*Failed to find vectorized operand*/
    auto vectorValueToStore = vectorValue->second;
    auto subViewOp = subViewBuilder.create<mlir::SubViewOp>(
                 forOp.getLoc(), targetMemref, memoryOp.getMemRef());
    auto newStoreOp = b.create<mlir::AffineStoreOp>(
        opInst->getLoc(), vectorValueToStore, subViewOp.getResult(), indices);
    operationsToDelete.push(opInst);
    vectorOperations.push(subViewOp);
    vectorOperations.push(newStoreOp);
  }
  if (auto load = dyn_cast<AffineLoadOp>(opInst)) {
    auto lastIndex = load.getOperand(load.getNumOperands() - 1);
    auto isInvariant = isAccessInvariant(iv, lastIndex);
    if(isInvariant){
      auto newSplatOp = b.create<mlir::SplatOp>(
          load.getLoc(), vectorType, load.getResult());
      operandReplacement.insert(std::make_pair(load.getResult(), newSplatOp.getResult()));
      opInst->moveBefore(newSplatOp);
	  vectorOperations.push(newSplatOp);
    }
    else{
      auto subViewOp = subViewBuilder.create<mlir::SubViewOp>(
          opInst->getLoc(), targetMemref, memoryOp.getMemRef());
      auto newLoadOp = b.create<mlir::AffineLoadOp>(
          opInst->getLoc(), subViewOp.getResult(), indices);
      operandReplacement.insert(std::make_pair(opInst->getResult(0), newLoadOp.getOperation()->getResult(0)));
      operationsToDelete.push(opInst);
      vectorOperations.push(subViewOp);
      vectorOperations.push(newLoadOp);
    }
   
  }
  return success();
}



static PassRegistration<VectorizeOuterLoop> pass("affine-outerloop-vectorize",
                                               "Vectorize outer loop");
