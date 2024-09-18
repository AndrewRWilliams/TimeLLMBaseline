import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt

from models.TimeLLM import Model as TimeLLMModel

from typing import Optional, Union, Dict, Callable, Iterable

# CHECK FOR DEVICE 
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch_device}")

# check available device memory
if torch_device.type == "cuda":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0)}")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0)}")

def truncate_mse_loss(future_time, future_pred):
    # Assumes future_time.shape == (B, T1) and future_pred.shape == (B, T2)
    min_length = min(future_time.shape[-1], future_pred.shape[-1])
    return F.mse_loss(future_time[...,:min_length], future_pred[...,:min_length])

def truncate_mae_loss(future_time, future_pred):
    # Assumes future_time.shape == (B, T1) and future_pred.shape == (B, T2)
    min_length = min(future_time.shape[-1], future_pred.shape[-1])
    return F.l1_loss(future_time[...,:min_length], future_pred[...,:min_length])

class TimeLLMStarCasterWrapper(nn.Module):

    def __init__(self, time_llm_model):
        super().__init__()

        assert isinstance(time_llm_model, TimeLLMModel), f"TimeLLMStarCasterWrapper can only wrap a model of class TimeLLM.Model but got {type(time_llm_model)}"
        self.time_llm_model = time_llm_model
    
    def forward(self, past_time, context):
        self.time_llm_model.description = context
        return self.time_llm_model(x_enc=past_time.unsqueeze(-1), x_mark_enc=None, x_dec=None, x_mark_dec=None).squeeze(-1)

class EvaluationPipeline:

    def __init__(
        self, 
        model: TimeLLMModel, 
        metrics: Optional[Union[Callable, Dict[str, Callable]]] = None
    ):
        # self.dataset = dataset
        self.model = TimeLLMStarCasterWrapper(model)
        self.metrics = metrics if metrics is not None else {"mse_loss" : truncate_mse_loss}
    
    # TODO: This method needs to be replaced to handle actual StarCaster benchmark
    def get_evaluation_loader(self) -> Iterable:
        samples = []
        for sample in self.dataset.values():
            past_time = torch.from_numpy(sample["past_time"].to_numpy().T).float()
            future_time = torch.from_numpy(sample["future_time"].to_numpy().T).float()
            context = sample["context"]

            samples.append([past_time, future_time, context])

        return samples

    def compute_loss(self, future_time, future_pred): # TODO: Add support for multiple metrics
        return {m_name : m(future_time, future_pred) for m_name, m in self.metrics.items()}
    
    def evaluation_step(self, past_time, future_time, context):
        future_pred = self.model(past_time, context)
        loss = self.compute_loss(future_time, future_pred)
        return loss, future_pred
            
    @torch.no_grad()
    def eval(self):
        assert isinstance(self.model, TimeLLMStarCasterWrapper), "EvaluationPipeline currently only supports testing the TimeLLM architecture"

        infer_dataloader = self.get_evaluation_loader()
        losses, predictions = {m_name : [] for m_name in self.metrics.keys()}, []
        for past_time, future_time, context in infer_dataloader:
            loss_dict, preds = self.evaluation_step(past_time, future_time, context)

            for m_name, loss in loss_dict.items(): losses[m_name].append(loss)
            predictions.append(preds)

        return losses, predictions

class TimeLLMPredictor:

    def __init__(self,
    ckpt_path: Optional[str] = None):
        from argparse import ArgumentParser

        parser = ArgumentParser(description="Time-LLM")
        # basic config
        parser.add_argument(
            "--task_name",
            type=str,
            default="long_term_forecast",
            help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
        )
        parser.add_argument("--is_training", type=int, default=0, help="status")
        parser.add_argument("--model_id", type=str, default="test", help="model id")
        parser.add_argument(
            "--model_comment",
            type=str,
            default="none",
            help="prefix when saving test results",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="TimeLLM",
            help="model name, options: [Autoformer, DLinear]",
        )
        parser.add_argument("--seed", type=int, default=2021, help="random seed")
        parser.add_argument("--resume", action="store_true", default=False)

        # data loader
        parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
        parser.add_argument(
            "--root_path", type=str, default="./dataset", help="root path of the data file"
        )
        parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
        parser.add_argument(
            "--features",
            type=str,
            default="M",
            help="forecasting task, options:[M, S, MS]; "
            "M:multivariate predict multivariate, S: univariate predict univariate, "
            "MS:multivariate predict univariate",
        )
        parser.add_argument(
            "--target", type=str, default="OT", help="target feature in S or MS task"
        )
        parser.add_argument("--loader", type=str, default="modal", help="dataset type")
        parser.add_argument(
            "--freq",
            type=str,
            default="h",
            help="freq for time features encoding, "
            "options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], "
            "you can also use more detailed freq like 15min or 3h",
        )
        parser.add_argument(
            "--checkpoints",
            type=str,
            default="./checkpoints/",
            help="location of model checkpoints",
        )

        # forecasting task
        parser.add_argument("--seq_len", type=int, default=384, help="input sequence length")
        parser.add_argument("--label_len", type=int, default=48, help="start token length")
        parser.add_argument(
            "--pred_len", type=int, default=96, help="prediction sequence length"
        )
        parser.add_argument(
            "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
        )

        # model define
        parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
        parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
        parser.add_argument("--c_out", type=int, default=7, help="output size")
        parser.add_argument("--d_model", type=int, default=32, help="dimension of model")
        parser.add_argument("--n_heads", type=int, default=32, help="num of heads")
        parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
        parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
        parser.add_argument("--d_ff", type=int, default=32, help="dimension of fcn")
        parser.add_argument(
            "--moving_avg", type=int, default=25, help="window size of moving average"
        )
        parser.add_argument("--factor", type=int, default=1, help="attn factor")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        parser.add_argument(
            "--embed",
            type=str,
            default="timeF",
            help="time features encoding, options:[timeF, fixed, learned]",
        )
        parser.add_argument("--activation", type=str, default="gelu", help="activation")
        parser.add_argument(
            "--output_attention",
            action="store_true",
            help="whether to output attention in encoder",
        )
        parser.add_argument("--patch_len", type=int, default=16, help="patch length")
        parser.add_argument("--stride", type=int, default=8, help="stride")
        parser.add_argument(
            "--prompt_domain", type=int, default=0, help=""
        )  # TODO: change to 1
        parser.add_argument(
            "--llm_model", type=str, default="LLAMA", help="LLM model"
        )  # LLAMA, GPT2, BERT
        parser.add_argument(
            "--llm_dim", type=int, default="4096", help="LLM model dimension"
        )  # LLama7b:4096; GPT2-small:768; BERT-base:768

        # optimization
        parser.add_argument(
            "--num_workers", type=int, default=10, help="data loader num workers"
        )
        parser.add_argument("--itr", type=int, default=1, help="experiments times")
        parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
        parser.add_argument("--align_epochs", type=int, default=10, help="alignment epochs")
        parser.add_argument(
            "--batch_size", type=int, default=16, help="batch size of train input data"
        )
        parser.add_argument(
            "--eval_batch_size", type=int, default=8, help="batch size of model evaluation"
        )
        parser.add_argument(
            "--patience", type=int, default=10, help="early stopping patience"
        )
        parser.add_argument(
            "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
        )
        parser.add_argument("--des", type=str, default="test", help="exp description")
        parser.add_argument("--loss", type=str, default="MSE", help="loss function")
        parser.add_argument(
            "--lradj", type=str, default="type1", help="adjust learning rate"
        )
        parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")
        parser.add_argument(
            "--use_amp",
            action="store_true",
            help="use automatic mixed precision training",
            default=False,
        )
        parser.add_argument("--llm_layers", type=int, default=32)
        parser.add_argument("--percent", type=int, default=100)

        # StarCaster Pipeline
        # parser.add_argument("data_path", type=str, required=True)
        parser.add_argument("--ckpt_path", type=str, default=None)

        args = parser.parse_args()

        self.model = TimeLLMModel(args).to(torch_device)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(
                ckpt["module"]
            )  # TODO: Change this to not be specific to the Time-LLM checkpoint
        else:
            raise ValueError("No checkpoint path provided")


    def __call__(
        self,
        context: str,
        past_time: torch.Tensor,
        future_time: torch.Tensor,
        ):

        pipeline = EvaluationPipeline(
            self.model,
            metrics={"mse_loss": truncate_mse_loss, "mae_loss": truncate_mae_loss},
        )
        losses, predictions = pipeline.evaluation_step(past_time, future_time, context)
        # print(f"Got losses: {losses}")
        # print(f"Predictions has shape: {[pred.shape for pred in predictions]}")
        
        return predictions



if __name__ == "__main__":
    predictor = TimeLLMPredictor(
        "/home/toolkit/hf_download/Time-LLM-ETTh1-pl_96-ckpt/pytorch_model/mp_rank_00_model_states.pt"
    )
    # Generate time steps for the sine wave
    time_steps = torch.linspace(0, 2 * torch.pi, 96).unsqueeze(0).to(torch_device)

    # Generate sine wave for past_time and future_time
    past_time = torch.sin(time_steps).to(torch_device)  # Sine wave for past time
    future_time = torch.sin(time_steps + torch.pi).to(torch_device)  # Sine wave phase-shifted for future time
    context = "straight line upwards"
    predictions = predictor(
        context =context,
        past_time = past_time,
        future_time = future_time

    )
    # Plot
    fig, ax = plt.subplots()

    # Concatenate past and future time and plot
    concat_time = torch.cat([past_time, future_time], dim=-1)
    ax.plot(concat_time.squeeze().cpu().numpy(), label="Past + Future Time", color='blue')

    # Plot predictions at the same steps as future_time
    ax.plot(range(96, 96 + predictions.size(-1)), predictions.detach().squeeze().cpu().numpy(), label="Predictions", color='orange')

    # Add a vertical line to separate past and future time
    ax.axvline(x=96, color='black', linestyle='--', label='Now')

    # Add context as text above the plot
    ax.text(0.5, 1.05, f"Context: {context}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Add legend and labels
    ax.legend()
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Values")

    # Save the plot
    plt.savefig("/home/toolkit/Time-LLM/test.png")
