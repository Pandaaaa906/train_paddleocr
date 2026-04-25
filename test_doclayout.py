from paddlex import create_model

model_name = "PP-DocLayoutV3"
model = create_model(model_name=model_name)
output = model.predict("layout_analysis_demo.jpg", batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")




if __name__ == '__main__':
    pass