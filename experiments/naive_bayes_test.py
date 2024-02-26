from train_classifier import load_dataset_from_csv, test_naive_bayes
import concurrent.futures

# A short program I created to test the accuracy of the Naive Bayes calssifier over multiple tests, using multiple test-train split values

def test(sample: float, size: int) -> float:
    print(f"Starting {sample}...")
    scores = []
    for i in range(0,size,1):
        dataset = load_dataset_from_csv(size=sample)[1]
        scores.append(test_naive_bayes(dataset))
    return sum(scores)/len(scores)

def main(): 
    size = 1000
    sample = [0.1,0.15,0.2,0.25]
    scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sample), 10)) as executor:
        future_scores = [executor.submit(test, s, size) for s in sample]
        for score in future_scores:
            scores.append(score.result())
    for i in range(0,len(scores),1):
        print(f"Average for {sample[i]:.2f}: {scores[i]:.3f} over {size} tests")

if __name__ == "__main__": main()