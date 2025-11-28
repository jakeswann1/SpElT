# SessionManager Guide

## Overview

The `SessionManager` class provides a clean, efficient way to process multiple ephys sessions without repetitive for-loops. It enables:

- **Parallel processing** across multiple CPU cores
- **Cleaner code** with function-based analysis instead of nested loops
- **Easy filtering** of session subsets
- **Automatic result aggregation** into pandas DataFrames

## Why Use SessionManager?

### Before (Old Approach)
```python
# Repetitive for-loop in every notebook
session_dict = load_sessions_from_config('config.yaml', 'all_neuropixels')
results = []

for i, session_path in enumerate(session_dict.values()):
    print(f"Processing {i+1}/{len(session_dict)}")
    obj = ephys(path=session_path, sheet_url=sheet_path)
    obj._load_ephys(keep_good_only=True)

    # Your custom analysis here
    unit_count = obj.analyzer.get_num_units()

    results.append({
        'session': session_path,
        'units': unit_count
    })

df = pd.DataFrame(results)
```

**Problems:**
- Runs serially (slow!)
- Repetitive setup code in every notebook
- Hard to reuse analysis across notebooks
- Difficult to debug

### After (SessionManager Approach)
```python
# Initialize once
manager = SessionManager(
    config_path='config.yaml',
    config_key='all_neuropixels',
    sheet_path=sheet_path
)

# Define analysis function (reusable!)
def analyze_units(session_name, obj):
    obj._load_ephys(keep_good_only=True)
    return {
        'session': session_name,
        'units': obj.analyzer.get_num_units()
    }

# Run in parallel across all sessions
df = manager.map_to_dataframe(analyze_units, n_jobs=8)
```

**Benefits:**
- Runs in parallel (8x faster with 8 cores!)
- Clean, reusable functions
- Easy to test and debug
- One-line execution

## Basic Usage

### 1. Initialize SessionManager

```python
from spelt.session_manager import SessionManager

manager = SessionManager(
    config_path='../session_selection.yaml',
    config_key='all_neuropixels',
    sheet_path='https://docs.google.com/spreadsheets/...'
)

print(f"Loaded {len(manager)} sessions")
```

### 2. Define Your Analysis Function

Your function should:
- Take `session_name` and `obj` (ephys object) as arguments
- Return a dict with results (for easy DataFrame conversion)

```python
def my_analysis(session_name, obj):
    """Analyze a single session."""
    obj._load_ephys(keep_good_only=True)

    # Do your analysis
    n_units = obj.analyzer.get_num_units()

    return {
        'session': session_name,
        'animal': obj.animal,
        'date': obj.date,
        'n_units': n_units
    }
```

### 3. Run Across All Sessions

```python
# Returns a DataFrame automatically
df = manager.map_to_dataframe(
    my_analysis,
    n_jobs=8,           # Use 8 CPU cores (-1 for all cores)
    show_progress=True  # Show progress bar
)
```

## Advanced Usage

### Passing Additional Parameters

```python
def analyze_with_params(session_name, obj, min_units=5, sampling_rate=1000):
    obj._load_ephys(keep_good_only=True)
    n_units = obj.analyzer.get_num_units()

    if n_units < min_units:
        return None  # Skip this session

    # Use the parameters
    obj.load_lfp([0], sampling_rate=sampling_rate, channels=[0])

    return {'session': session_name, 'n_units': n_units}

# Pass extra parameters as kwargs
results = manager.map_to_dataframe(
    analyze_with_params,
    n_jobs=4,
    min_units=10,        # Passed to function
    sampling_rate=500    # Passed to function
)
```

### Filtering Sessions

```python
# Only process sessions from specific animals
results = manager.map_to_dataframe(
    my_analysis,
    n_jobs=4,
    filter_sessions=lambda s: 'r1572' in s or 'r1672' in s
)

# Or create a subset manager
r1572_manager = manager.get_session_subset(lambda s: 'r1572' in s)
results = r1572_manager.map_to_dataframe(my_analysis, n_jobs=4)
```

### Per-Unit Analysis (Multiple Rows Per Session)

```python
def analyze_all_units(session_name, obj):
    """Return a list of dicts, one per unit."""
    obj._load_ephys(keep_good_only=True)

    results = []
    for unit_id in obj.analyzer.unit_ids:
        results.append({
            'session': session_name,
            'unit_id': unit_id,
            # ... more metrics
        })

    return results

# Use map() instead of map_to_dataframe()
all_results = manager.map(analyze_all_units, n_jobs=8)

# Flatten the list of lists
flattened = [item for sublist in all_results for item in sublist]
df = pd.DataFrame(flattened)
```

### Error Handling

```python
def robust_analysis(session_name, obj):
    """Handle errors gracefully."""
    try:
        obj._load_ephys(keep_good_only=True)
        # ... analysis that might fail
        return {'session': session_name, 'status': 'success'}
    except Exception as e:
        return {'session': session_name, 'status': 'failed', 'error': str(e)}

df = manager.map_to_dataframe(robust_analysis, n_jobs=4)

# Separate successful and failed
successful = df[df['status'] == 'success']
failed = df[df['status'] == 'failed']
```

### Serial Processing for Debugging

```python
# Use n_jobs=1 to see full error traces
df = manager.map_to_dataframe(my_analysis, n_jobs=1, show_progress=True)
```

## Performance Tips

1. **Choose the right n_jobs:**
   - `n_jobs=1`: Serial processing (for debugging)
   - `n_jobs=4`: Use 4 cores
   - `n_jobs=-1`: Use all available cores
   - `n_jobs=-2`: Use all cores except 1 (leaves one core free)

2. **Don't load unnecessary data:**
   - Only load what you need (spikes, LFP, position)
   - Use `sparse=True` when possible

3. **Use caching for repeated access:**
   ```python
   manager = SessionManager(..., cache_objects=True)  # Keeps objects in RAM
   ```
   (Only do this if you have enough RAM!)

4. **Profile your analysis function:**
   ```python
   import time

   def my_analysis(session_name, obj):
       start = time.time()
       # ... your analysis
       print(f"{session_name}: {time.time() - start:.2f}s")
       return {...}
   ```

## Common Patterns

### Pattern 1: Unit Yield Analysis
```python
def unit_yield(session_name, obj, subjects_data):
    obj._load_ephys(keep_good_only=True)
    subject = subjects_data[subjects_data['ID'] == obj.animal]

    return {
        'animal': obj.animal,
        'date': obj.date,
        'n_units': obj.analyzer.get_num_units(),
        'age': obj.age
    }

df = manager.map_to_dataframe(unit_yield, n_jobs=8, subjects_data=subjects_df)
```

### Pattern 2: LFP Analysis
```python
def lfp_analysis(session_name, obj, sampling_rate=1000):
    obj._load_ephys(keep_good_only=True)
    obj.good_channels = np.unique(obj.analyzer.sorting.get_property('ch'))
    obj.load_lfp([0], sampling_rate, channels=obj.good_channels)

    # Compute power spectrum, etc.
    return {...}
```

### Pattern 3: Spike-Phase Coupling
```python
def spike_phase(session_name, obj):
    obj._load_ephys(keep_good_only=True, sparse=False)
    obj.load_lfp(obj.trial_iterators, 250, channels=obj.good_channels)
    obj.load_pos(obj.trial_iterators)

    # Compute phase locking for all units
    results = []
    for unit_id in obj.analyzer.unit_ids:
        # ... compute Rayleigh vector, etc.
        results.append({...})

    return results
```

## Migration Guide

To convert your existing notebook:

1. **Replace session loading loop:**
   ```python
   # OLD
   session_dict = load_sessions_from_config(...)
   for session_path in session_dict.values():
       obj = ephys(path=session_path, ...)

   # NEW
   manager = SessionManager(config_path=..., config_key=..., sheet_path=...)
   ```

2. **Extract loop body into a function:**
   - Take the code inside your loop
   - Put it in a function that takes `(session_name, obj)`
   - Return a dict with results

3. **Use map_to_dataframe:**
   ```python
   df = manager.map_to_dataframe(your_function, n_jobs=8)
   ```

## Examples

See these notebooks for complete examples:
- `ephys_analysis/unit-yield/Unit Yield Per Day - Refactored.ipynb`
- `ephys_analysis/examples & testing/SessionManager Examples.ipynb`

## API Reference

### SessionManager

**Constructor:**
```python
SessionManager(config_path, config_key, sheet_path, subject_sheet_path=None, cache_objects=False)
```

**Methods:**
- `map(func, n_jobs=1, show_progress=True, filter_sessions=None, **func_kwargs)` → List
- `map_to_dataframe(func, n_jobs=1, show_progress=True, filter_sessions=None, **func_kwargs)` → DataFrame
- `get_session_subset(filter_func)` → SessionManager
- `get_ephys_object(session_name)` → ephys object

**Attributes:**
- `session_dict`: Dict mapping session names to paths
- `session_names`: List of session names
- `session_paths`: List of session paths
