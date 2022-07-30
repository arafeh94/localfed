import wandb


wandb.init(project="test_project", entity="mwazzeh")
wandb.login(key="24db2a5612aaf7311dd29a5178f252a1c0a351a9")


wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}

acc = 0.4
loss = 0

model = None


thresh = 0.5

# wandb.alert(
#     title="Low accuracy",
#     text=f"Accuracy {acc} is below the acceptable threshold {thresh}"
# )



# Define the names of the columns in your Table
column_names = ["image_id", "image", "label", "prediction"]
# Prepare your data, row-wise
# You can log filepaths or image tensors with wandb.Image
images_data = [
    ['img_0.jpg', wandb.Image("children/images/clients_data_distribution.png"), 0, 0],
    ['img_1.jpg', wandb.Image("children/images/clients_data_distribution.png"), 8, 0],
    ['img_2.jpg', wandb.Image("children/images/clients_data_distribution.png"), 7, 1],
    ['img_3.jpg', wandb.Image("children/images/clients_data_distribution.png"), 1, 1]
]

# Create your W&B Table
val_table = wandb.Table(data=images_data, columns=column_names)

# Log the Table to W&B
# wandb.log({'my_val_table': val_table})
wandb.log({"acc": acc, "loss": loss, 'my_val_table': val_table})


# wandb.config.dropout = 0.2
# Optional
# wandb.watch(model)
