/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>

#include "torch/script.h"
#include "neural/factory.h"
#include "utils/bititer.h"

namespace lczero {
namespace {


class PytorchNetworkComputation;

class PytorchNetwork : public Network {
 public:
  PytorchNetwork(const OptionsDict& options)
      : capabilities_{
            static_cast<pblczero::NetworkFormat::InputFormat>(
                options.GetOrDefault<int>(
                    "input_mode",
                    pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE)),
            pblczero::NetworkFormat::MOVES_LEFT_NONE} {
    try {
      module_ = torch::jit::load("traced_model.pt");
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
    }
    module_.to(at::kCUDA);

  }

  std::unique_ptr<NetworkComputation> NewComputation() override;

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  torch::IValue Compute(torch::Tensor& input);

 private:
  torch::jit::script::Module module_;
  NetworkCapabilities capabilities_{
      // TODO depend on network structure
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
      pblczero::NetworkFormat::MOVES_LEFT_NONE};
};

torch::IValue PytorchNetwork::Compute(torch::Tensor& input) {
  std::vector<torch::IValue> inputs;
  inputs.push_back(input);
  return module_.forward(inputs);
}


class PytorchNetworkComputation : public NetworkComputation {
 public:
  PytorchNetworkComputation(PytorchNetwork* network): network_(network) {}

  void AddInput(InputPlanes&& input) override {
    raw_input_.emplace_back(input);
  }

  void ComputeBlocking() override {
    PrepareInput();
    auto output = network_->Compute(input_);
    auto elements = output.toTuple()->elements();
    policy_ = elements[0].toTensor().to(at::kCPU);
    value_ = elements[1].toTensor().to(at::kCPU);
  }

  int GetBatchSize() const override { return raw_input_.size(); }

  float GetQVal(int sample) const override {
      auto w = value_[sample][0].template item<float>();
      auto l = value_[sample][2].template item<float>();
      return w - l;
  }

  float GetDVal(int sample) const override {
      return value_[sample][1].template item<float>();
  }

      // TODO
  float GetMVal(int /* sample */) const override { return 0.0f; }

  float GetPVal(int sample, int move_id) const override {
      return policy_[sample][move_id].template item<float>();
  }

 private:
  void PrepareInput();

  std::vector<InputPlanes> raw_input_;
  PytorchNetwork* network_;
  torch::Tensor input_;
  torch::Tensor policy_;
  torch::Tensor value_;
};

std::unique_ptr<NetworkComputation> PytorchNetwork::NewComputation() {
    return std::make_unique<PytorchNetworkComputation>(this);
  }

void PytorchNetworkComputation::PrepareInput() {
  input_ = torch::zeros(
          {static_cast<int>(raw_input_.size()), 112, 8, 8});
  // float tensor with 4 dims
  auto input_a = input_.accessor<float, 4>();

  int sample_num = 0;
  for (const auto& sample : raw_input_) {
    CHECK_EQ(sample.size(), 112);
    int dim = 0;
    for (const auto& plane : sample) {
      for (auto bit : IterateBits(plane.mask)) {
        int r = bit / 8;
        int c = bit % 8;
        input_a[sample_num][dim][r][c] = static_cast<float>(plane.value);
//        std::cout << "writing " << sample_num << " " << dim << " (" << r << ", " << c << ") = " << plane.value << "\n";
      }
      dim++;
    }
    sample_num++;
  }
  input_ = input_.to(at::kCUDA);
}


}  // namespace

std::unique_ptr<Network> MakePytorchNetwork(
    const std::optional<WeightsFile>& /*weights*/, const OptionsDict& options) {
  return std::make_unique<PytorchNetwork>(options);
}

REGISTER_NETWORK("pytorch", MakePytorchNetwork, 0)

}  // namespace lczero

