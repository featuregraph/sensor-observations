# observation-models

Observation Models is a small experimental library for turning noisy sensor observations into probabilistic beliefs about hidden physical states.

The first version uses a Gaussian mixture model to convert measurements into evidence, then updates a persistent prior belief over possible states.

## Core idea

```text
raw measurement
      ↓
observation model
      ↓
probability vector
      ↓
prior update
      ↓
state prediction
```

## Example

```python
import numpy as np

from observation_models.models.sensors import Model
from observation_models.belief.prior import Prior
from observation_models.data.load_data import load_observations

train = load_observations(problem="rod", dataset="train")
test = load_observations(problem="rod", dataset="test")

X_test = np.diff(test[:, 1], prepend=0).reshape(-1, 1)

model = Model(train)
states, evidence = model.predict(X_test[:10])

prior = Prior(states, alpha=0.2)

for row in evidence:
    result = prior.update(row)
    print(prior.predict(), prior.confidence())
```

## Components

### `Model`

Fits an observation model from labeled training data.

Input training data is expected to have:

```text
[state, measurement]
```

The model returns:

```python
states, probabilities = model.predict(X)
```

where each row of `probabilities` is evidence over the possible states.

### `Prior`

Maintains a belief vector over possible states.

```python
prior = Prior(states, alpha=0.2)
prior.update(evidence)
```

The prior smooths noisy observations using an exponential moving update:

```text
belief = (1 - alpha) * belief + alpha * evidence
```

## Benchmark problems

The library is designed around simple physical systems where the true hidden state is known and sensor observations can be simulated.

Current benchmark direction:

- noisy physical measurements
- recursive belief updates
- future sensor fault simulation

## Project goals

Observation Models is meant to explore:

- noisy sensor interpretation
- observation-to-state inference
- recursive belief updates
- robust priors
- sensor fault detection
- reusable benchmark problems for physical systems

## Design principle

Observation models and belief models are separate.

The observation model answers:

> What does this measurement suggest?

The belief model answers:

> Given everything seen so far, what do we currently believe?

This separation allows different observation models and prior update rules to be tested independently.
