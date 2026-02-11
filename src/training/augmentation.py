"""
NLP Data Augmentation for CV Classifier.

Techniques:
1. Synonym Replacement - Replace words with WordNet synonyms
2. Random Deletion - Randomly remove words
3. Random Swap - Swap positions of random words
4. Random Insertion - Insert synonyms at random positions
"""

import random
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import nltk
from nltk.corpus import wordnet

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class TextAugmenter:
    """
    NLP-based text augmentation for balancing training data.

    Uses 4 techniques:
    - Synonym replacement
    - Random deletion
    - Random swap
    - Random insertion
    """

    def __init__(
        self,
        random_state: int = 42,
        min_word_length: int = 3
    ):
        """
        Args:
            random_state: Random seed for reproducibility
            min_word_length: Minimum word length for augmentation operations
        """
        self.random_state = random_state
        self.min_word_length = min_word_length
        random.seed(random_state)
        np.random.seed(random_state)

        self.stats: Dict[str, Any] = {}

    def get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.

        Args:
            word: Word to find synonyms for

        Returns:
            List of synonyms (may be empty)
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)

    def synonym_replacement(self, text: str, n: int = 3) -> str:
        """
        Replace n words with their synonyms.

        Args:
            text: Input text
            n: Number of words to replace

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 3:
            return text

        new_words = words.copy()

        # Get words that can be replaced (long enough)
        replaceable = [w for w in words if len(w) >= self.min_word_length]
        random.shuffle(replaceable)

        num_replaced = 0
        for word in replaceable:
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        return ' '.join(new_words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p.

        Args:
            text: Input text
            p: Probability of deleting each word

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) <= 1:
            return text

        # Keep words with probability (1 - p)
        new_words = [word for word in words if random.random() > p]

        # Ensure at least one word remains
        if len(new_words) == 0:
            return random.choice(words)

        return ' '.join(new_words)

    def random_swap(self, text: str, n: int = 2) -> str:
        """
        Swap n pairs of words randomly.

        Args:
            text: Input text
            n: Number of swaps to perform

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text

        new_words = words.copy()
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return ' '.join(new_words)

    def random_insertion(self, text: str, n: int = 2) -> str:
        """
        Insert n synonyms at random positions.

        Args:
            text: Input text
            n: Number of insertions

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text

        new_words = words.copy()

        # Get words that can provide synonyms
        source_words = [w for w in words if len(w) >= self.min_word_length]

        inserted = 0
        max_attempts = n * 3
        attempts = 0

        while inserted < n and attempts < max_attempts:
            attempts += 1
            if not source_words:
                break

            word = random.choice(source_words)
            synonyms = self.get_synonyms(word)

            if synonyms:
                synonym = random.choice(synonyms)
                insert_pos = random.randint(0, len(new_words))
                new_words.insert(insert_pos, synonym)
                inserted += 1

        return ' '.join(new_words)

    def augment(self, text: str, technique: str = 'random') -> str:
        """
        Apply a single augmentation technique.

        Args:
            text: Input text
            technique: One of 'synonym', 'delete', 'swap', 'insert', 'random'

        Returns:
            Augmented text
        """
        if technique == 'random':
            technique = random.choice(['synonym', 'delete', 'swap', 'insert'])

        if technique == 'synonym':
            return self.synonym_replacement(text)
        elif technique == 'delete':
            return self.random_deletion(text)
        elif technique == 'swap':
            return self.random_swap(text)
        elif technique == 'insert':
            return self.random_insertion(text)
        else:
            return text

    def augment_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = 'balance',
        target_per_class: Optional[int] = None,
        max_augment_per_sample: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment the entire dataset.

        Args:
            X: Array of texts
            y: Array of labels
            strategy: 'balance' (equalize classes) or 'multiply' (augment all)
            target_per_class: Target samples per class (for 'balance')
            max_augment_per_sample: Maximum augmentations per original sample

        Returns:
            Tuple of (X_augmented, y_augmented) - combined with original
        """
        print("\n" + "=" * 60)
        print(" DATA AUGMENTATION")
        print("=" * 60)

        class_counts = Counter(y)
        print(f"\nOriginal class distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
            print(f"   {cls}: {count}")

        if strategy == 'balance':
            return self._augment_balance(
                X, y, target_per_class, max_augment_per_sample, class_counts
            )
        elif strategy == 'multiply':
            return self._augment_multiply(X, y, max_augment_per_sample)
        else:
            print(f"Unknown strategy: {strategy}")
            return X, y

    def _augment_balance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_per_class: Optional[int],
        max_augment_per_sample: int,
        class_counts: Counter
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes through augmentation."""

        # Determine target
        if target_per_class is None:
            target_per_class = int(np.mean(list(class_counts.values())))

        print(f"\nTarget samples per class: {target_per_class}")
        print(f"Max augmentations per sample: {max_augment_per_sample}")

        augmented_X = []
        augmented_y = []
        techniques_used = Counter()

        for cls in class_counts:
            current_count = class_counts[cls]
            needed = max(0, target_per_class - current_count)

            if needed == 0:
                continue

            # Get samples of this class
            class_mask = y == cls
            class_samples = X[class_mask]

            # Calculate augmentations needed per sample
            augments_needed = min(needed, len(class_samples) * max_augment_per_sample)

            print(f"\n   {cls}: {current_count} -> {current_count + augments_needed} (+{augments_needed})")

            for i in range(augments_needed):
                # Pick a random sample from this class
                sample = random.choice(class_samples)

                # Pick a random technique
                technique = random.choice(['synonym', 'delete', 'swap', 'insert'])
                techniques_used[technique] += 1

                # Augment
                augmented_text = self.augment(sample, technique)

                augmented_X.append(augmented_text)
                augmented_y.append(cls)

        print(f"\n   Total augmented samples: {len(augmented_X)}")
        print(f"   Techniques used: {dict(techniques_used)}")

        # Combine with original
        if len(augmented_X) > 0:
            X_final = np.concatenate([X, np.array(augmented_X)])
            y_final = np.concatenate([y, np.array(augmented_y)])
        else:
            X_final = X
            y_final = y

        # Stats
        self.stats = {
            'original_samples': len(X),
            'augmented_samples': len(augmented_X),
            'final_samples': len(X_final),
            'target_per_class': target_per_class,
            'techniques_used': dict(techniques_used)
        }

        print(f"\n   Final dataset: {len(X_final)} samples")

        return X_final, y_final

    def _augment_multiply(
        self,
        X: np.ndarray,
        y: np.ndarray,
        multiplier: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment all samples by a multiplier."""

        print(f"\nMultiplying all samples by {multiplier}...")

        augmented_X = list(X)
        augmented_y = list(y)
        techniques_used = Counter()

        for i, (text, label) in enumerate(zip(X, y)):
            for _ in range(multiplier - 1):  # -1 because original already counted
                technique = random.choice(['synonym', 'delete', 'swap', 'insert'])
                techniques_used[technique] += 1

                augmented_text = self.augment(text, technique)
                augmented_X.append(augmented_text)
                augmented_y.append(label)

        self.stats = {
            'original_samples': len(X),
            'augmented_samples': len(augmented_X) - len(X),
            'final_samples': len(augmented_X),
            'multiplier': multiplier,
            'techniques_used': dict(techniques_used)
        }

        print(f"   Original: {len(X)}")
        print(f"   After augmentation: {len(augmented_X)}")
        print(f"   Techniques used: {dict(techniques_used)}")

        return np.array(augmented_X), np.array(augmented_y)


def demo_augmentation():
    """Demonstrate augmentation techniques."""
    augmenter = TextAugmenter(random_state=42)

    sample = "Experienced Python developer with machine learning skills and data analysis expertise"

    print("Original text:")
    print(f"  {sample}")
    print("\nAugmentations:")
    print(f"  Synonym:   {augmenter.synonym_replacement(sample)}")
    print(f"  Deletion:  {augmenter.random_deletion(sample)}")
    print(f"  Swap:      {augmenter.random_swap(sample)}")
    print(f"  Insertion: {augmenter.random_insertion(sample)}")


if __name__ == "__main__":
    demo_augmentation()
