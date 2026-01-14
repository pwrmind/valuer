import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def main():
    print("Hello from valuer!")
    # 1. Выбор базы (всего 80-100МБ в исходном виде, отлично понимает русский)
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Данные (формат: текст задачи и вектор оценок 0.0-1.0 для 3 ценностей)
    # Пример: {"text": "Сходить в спортзал", "labels": 
    # [ "Здоровье", "Энергия", "Настроение", "Сон", "Работа", "Финансы", "Обучение", "Проекты", "Семья", "Друзья", "Отношения", "Сообщество", "Творчество", "Хобби", "Духовность"]
    # [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.3, 0.0, 0.7, 0.8, 0.0, 0.9, 0.1, 0.0, 0.9]

    dataset = load_dataset("json", data_files="values_distilled.jsonl")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    # 3. Модель для регрессии (градации)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=15, 
        problem_type="multi_label_classification"
    ).to("cuda") # Обучение на GPU

    # 4. Настройка GPU-обучения
    args = TrainingArguments(
        output_dir="./value_model",
        per_device_train_batch_size=32,
        num_train_epochs=10,
        learning_rate=5e-5,
        fp16=True,             # Смешанная точность для ускорения на GPU
        save_total_limit=1,
        eval_strategy ="no"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds["train"])
    trainer.train()

    model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")


if __name__ == "__main__":
    main()
