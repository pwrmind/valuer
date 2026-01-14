def main():
    print("Hello from valuer!")
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    # 1. Используем MiniLM (быстрая, мультиязычная, ~110MB в исходном виде)
    base_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 2. Модель для градации (Regression/Multi-label)
    # Предположим, у нас 3 ценности: Семья, Карьера, Саморазвитие
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, 
        num_labels=15, 
        problem_type="multi_label_classification"
    ).to("cuda")

    # 3. Загрузка вашего синтетического датасета от LLM
    dataset = load_dataset("json", data_files="dataset.json")["train"]

    def tokenize_fn(batch):
        # Ограничиваем длину 128 токенами для скорости в браузере
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    # 4. Обучение на GPU
    args = TrainingArguments(
        output_dir="./training_results",
        per_device_train_batch_size=32,
        num_train_epochs=5,
        learning_rate=3e-5,
        fp16=True, # Ускорение на GPU
        save_strategy="no"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds)
    trainer.train()

    # 5. Сохранение (явно перезаписываем токенизатор, чтобы убрать баги Mistral)
    save_path = "./clean_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Модель сохранена в {save_path}")

if __name__ == "__main__":
    main()
