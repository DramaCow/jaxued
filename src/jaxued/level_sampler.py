from typing import Literal, Optional, TypedDict, Tuple
import chex
import jax
import jax.numpy as jnp

from jaxued.environments.underspecified_env import Level

Prioritization = Literal["rank", "topk"]

class Sampler(TypedDict):
    levels:         chex.Array # shape (capacity, ...)
    scores:         chex.Array # shape (capacity)
    timestamps:     chex.Array # shape (capacity)
    size:           int
    episode_count:  int
    level_extra:    Optional[dict]

class LevelSampler:
    """
    The `LevelSampler` provides all of the functionality associated with a level buffer in a PLR/ACCEL-type method. In the standard Jax style, the level sampler class does not store any data itself, and accepts a `sampler` object for most operations.

    Examples:
        >>>
        pholder_level       = ...
        pholder_level_extra = ...
        level_sampler       = LevelSampler(4000)
        sampler             = level_sampler.initialize(pholder_level, pholder_level_extra)
        should_replay       = level_sampler.sample_replay_decision(sampler, rng)
        replay_levels       = level_sampler.sample_replay_levels(sampler, rng, 32) # 32 replay levels
        scores              = ... # eval agent
        sampler             = level_sampler.insert_batch(sampler, level, scores)

    Args:
        capacity (int): The maximum number of levels that can be stored in the buffer.
        replay_prob (float, optional): The chance of performing on_replay vs on_new. Defaults to 0.95.
        staleness_coeff (float, optional): The weighting factor for staleness. Defaults to 0.5.
        minimum_fill_ratio (float, optional): The class will never sample a replay decision until the level buffer is at least as full as specified by this value. Defaults to 1.0.
        prioritization_params (dict, optional): If prioritization="rank", this has a "temperature" field; for "topk" it has a "k" field. If not provided, by default this is initialized to a temperature of 1.0 and k=1. Defaults to None.
        duplicate_check (bool, optional): If this is true, duplicate levels cannot be added to the buffer. This adds some computation to check for duplicates. Defaults to False.
    """
    def __init__(
        self,
        capacity: int,
        replay_prob: float = 0.95,
        staleness_coeff: float = 0.5,
        minimum_fill_ratio: float = 1.0, # minimum fill required before replay can occur
        prioritization: Prioritization = "rank",
        prioritization_params: dict = None,
        duplicate_check: bool = False,
    ):
        self.capacity = capacity
        self.replay_prob = replay_prob
        self.staleness_coeff = staleness_coeff
        self.minimum_fill_ratio = minimum_fill_ratio
        self.prioritization = prioritization
        self.prioritization_params = prioritization_params
        self.duplicate_check = duplicate_check
        
        if prioritization_params is None:
            if prioritization == "rank":
                self.prioritization_params = {"temperature": 1.0}
            elif prioritization == "topk":
                self.prioritization_params = {"k": 1}
            else:
                raise Exception(f"\"{prioritization}\" not a valid prioritization.")
        
    def initialize(self, pholder_level: Level, pholder_level_extra=None) -> Sampler:
        """
        Returns the `sampler` object as a dictionary.

        Sampler Object Keys:
            * "levels" (shape (self.capacity, ...)): the levels themselves
            * "scores" (shape (self.capacity)): the scores of the levels
            * "timestamps" (shape (self.capacity)): the timestamps of the levels
            * "size" (int): the number of levels currently in the buffer
            * "episode_count" (int): the number of episodes that have been played so far

        Args:
            pholder_level (Level): A placeholder level that will be used to initialize the level buffer.
            pholder_level_extra (dict, optional): If given, this should be a dictionary with arbitrary keys that is kept track of alongside each level. An example is "max_return" for each level. Defaults to None.

        Returns:
            Sampler: The initialized sampler object
        """
        sampler = {
            "levels": jax.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_level),
            "scores": jnp.full(self.capacity, -jnp.inf, dtype=jnp.float32),
            "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
            "size": 0,
            "episode_count": 0,
        }
        if pholder_level_extra is not None:
            sampler["levels_extra"] = jax.tree_map(lambda x: jnp.array([x]).repeat(self.capacity, axis=0), pholder_level_extra)
        return sampler
        
    def sample_replay_decision(self, sampler: Sampler, rng: chex.PRNGKey) -> bool:
        """
        Returns a single boolean indicating if a `replay` or `new` step should be taken. This is based on the proportion of the buffer that is filled and the `replay_prob` parameter.

        Args:
            sampler (Sampler): The sampler object
            rng (chex.PRNGKey): 

        Returns:
            bool: 
        """
        proportion_filled = self._proportion_filled(sampler)
        return (proportion_filled >= self.minimum_fill_ratio) & (jax.random.uniform(rng) < self.replay_prob)
    
    def sample_replay_level(self, sampler: Sampler, rng: chex.PRNGKey) -> Tuple[Sampler, Tuple[int, Level]]:
        """
        Samples a replay level from the buffer. It does this by first computing the weights of each level (using `level_weights`), and then sampling from the buffer using these weights. The `sampler` object is updated to reflect the new episode count and the level that was sampled. The level itself is returned as well as the index of the level in the buffer.

        Args:
            sampler (Sampler): The sampler object
            rng (chex.PRNGKey): 

        Returns:
            Tuple[Sampler, Tuple[int, Level]]: The updated sampler object, the sampled level's index and the level itself.
        """
        weights = self.level_weights(sampler)
        idx = jax.random.choice(rng, self.capacity, p=weights)
        new_episode_count = sampler["episode_count"] + 1
        sampler = {
            **sampler,
            "timestamps": sampler["timestamps"].at[idx].set(new_episode_count),
            "episode_count": new_episode_count,
        }
        return sampler, (idx, jax.tree_map(lambda x: x[idx], sampler["levels"]))
    
    def sample_replay_levels(self, sampler: Sampler, rng: chex.PRNGKey, num: int) -> Tuple[Sampler, Tuple[chex.Array, Level]]:
        """
        Samples several levels by iteratively calling `sample_replay_level`. The `sampler` object is updated to reflect the new episode count and the levels that were sampled.

        Args:
            sampler (Sampler): The sampler object
            rng (chex.PRNGKey): 
            num (int): How many levels to sample

        Returns:
            Tuple[Sampler, Tuple[chex.Array, Level]]: The updated sampler, an array of indices, and multiple levels.
        """
        return jax.lax.scan(self.sample_replay_level, sampler, jax.random.split(rng, num), length=num)
    
    def insert(self, sampler: Sampler, level: Level, score: float, level_extra: dict=None) -> Tuple[Sampler, int]:
        """
        Attempt to insert level into the level buffer.
        
        Insertion occurs when:
        - Corresponding score exceeds the score of the lowest weighted level
          currently in the buffer (in which case it will replace it).
        - Buffer is not yet at capacity.
        
        Optionally, if the level to be inserted already exists in the level
        buffer, the corresponding buffer entry will be updated instead.
        (See, `duplicate_check`)

        Args:
            sampler (Sampler): The sampler object
            level (Level): Level to insert
            score (float): Its score
            level_extra (dict, optional): If level extra was given in `initialize`, then it must be given here too. Defaults to None.

        Returns:
            Tuple[Sampler, int]: The updated sampler, and the level's index in the buffer (-1 if it was not inserted)
        """
        if self.duplicate_check:
            idx = self.find(sampler, level)
            return jax.lax.cond(
                idx == -1,
                lambda: self._insert_new(sampler, level, score, level_extra),
                lambda: ({
                    **self.update(sampler, idx, score, level_extra), # what happens to mutation rate here?
                    "timestamps": sampler["timestamps"].at[idx].set(sampler["episode_count"] + 1),
                    "episode_count": sampler["episode_count"] + 1
                }, idx),
            )
        return self._insert_new(sampler, level, score, level_extra)
    
    def insert_batch(self, sampler: Sampler, levels: Level, scores: chex.Array, level_extras: dict=None) -> Tuple[Sampler, chex.Array]:
        """
        Inserts a batch of levels.

        Args:
            sampler (Sampler): The sampler object
            levels (_type_): The levels to insert. This must be a `batched` level, in that it has an extra dimension at the front.
            scores (_type_): The scores of each level
            level_extras (dict, optional): The optional level_extras. Defaults to None.
        """
        def _insert(sampler, step):
            level, score, level_extra = step
            return self.insert(sampler, level, score, level_extra)
        return jax.lax.scan(_insert, sampler, (levels, scores, level_extras))
    
    def find(self, sampler: Sampler, level: Level) -> int:
        """
        Returns the index of level in the level buffer. If level is not present, -1 is returned.

        Args:
            sampler (Sampler): The sampler object
            level (Level): The level to find

        Returns:
            int: index or -1 if not found.
        """
        eq_tree = jax.tree_map(lambda X, y: (X == y).reshape(self.capacity, -1).all(axis=-1), sampler["levels"], level)
        eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
        eq_mask = jnp.array(eq_tree_flat).all(axis=0) & (jnp.arange(self.capacity) < sampler["size"])
        return jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)
    
    def get_levels(self, sampler: Sampler, level_idx: int) -> Level:
        """
        Returns the level at a particular index.

        Args:
            sampler (Sampler): The sampler object
            level_idx (int): The index to return

        Returns:
            Level: 
        """
        return jax.tree_map(lambda x: x[level_idx], sampler["levels"])
    
    def get_levels_extra(self, sampler: Sampler, level_idx: int) -> dict:
        """
        Returns the level extras associated with a particular index

        Args:
            sampler (Sampler): The sampler object
            level_idx (int): The index to return

        Returns:
            dict: 
        """
        return jax.tree_map(lambda x: x[level_idx], sampler["levels_extra"])
    
    def update(self, sampler: Sampler, idx: int, score: float, level_extra: dict=None) -> Sampler:
        """
        This updates the score and level_extras of a level

        Args:
            sampler (Sampler): The sampler object
            idx (int): The index of the level
            score (float): The score of the level
            level_extra (dict, optional): The associated. Defaults to None.

        Returns:
            Sampler: Updated Sampler
        """
        new_sampler = {
            **sampler,
            "scores": sampler["scores"].at[idx].set(score),
        }
        if level_extra is not None:
            new_sampler["levels_extra"] = jax.tree_map(lambda x, y: x.at[idx].set(y), new_sampler["levels_extra"], level_extra)
        return new_sampler
    
    def update_batch(self, sampler: Sampler, level_inds: chex.Array, scores: chex.Array, level_extras: dict=None) -> Sampler:
        """
        Updates the scores and level_extras of a batch of levels.

        Args:
            sampler (Sampler): The sampler object
            level_inds (chex.Array): Level indices
            scores (chex.Array): Scores
            level_extras (dict, optional): . Defaults to None.

        Returns:
            Sampler: Updated Sampler
        """
        def _update(sampler, step):
            level_idx, score, level_extra = step
            return self.update(sampler, level_idx, score, level_extra), None
        return jax.lax.scan(_update, sampler, (level_inds, scores, level_extras))[0]
        
    def level_weights(self, sampler: Sampler, prioritization: Prioritization=None, prioritization_params: dict=None) -> chex.Array:
        """
        Returns the weights for each level, taking into account both staleness and score.

        Args:
            sampler (Sampler): The sampler
            prioritization (Prioritization, optional): Possibly overrides self.prioritization. Defaults to None.
            prioritization_params (dict, optional): Possibly overrides self.prioritization_params. Defaults to None.

        Returns:
            chex.Array: Weights, shape (self.capacity)
        """
        w_s = self.score_weights(sampler, prioritization, prioritization_params)
        w_c = self.staleness_weights(sampler)
        return (1 - self.staleness_coeff) * w_s + self.staleness_coeff * w_c
    
    def score_weights(self, sampler: Sampler, prioritization: Prioritization=None, prioritization_params: dict=None) -> chex.Array:
        """
        Returns an array of shape (self.capacity) with the weights of each level (for sampling purposes).

        Args:
            sampler (Sampler): 
            prioritization (Prioritization, optional): Possibly overrides self.prioritization. Defaults to None.
            prioritization_params (dict, optional): Possibly overrides self.prioritization_params. Defaults to None.

        Returns:
            chex.Array: Score weights, shape (self.capacity)
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        
        if prioritization is None:
            prioritization = self.prioritization
        if prioritization_params is None:
            prioritization_params = self.prioritization_params
        
        if prioritization == "rank":
            ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()
            ranks = jnp.empty_like(ord).at[ord].set(jnp.arange(len(ord)) + 1)
            temperature = prioritization_params["temperature"]
            w_s = jnp.where(mask, 1 / ranks, 0) ** (1 / temperature)
            w_s = w_s / w_s.sum()
        elif prioritization == "topk":
            ord = (-jnp.where(mask, sampler["scores"], -jnp.inf)).argsort()
            k = prioritization_params["k"]
            topk_mask = jnp.empty_like(ord).at[ord].set(jnp.arange(self.capacity) < jnp.minimum(sampler["size"], k))
            w_s = jax.nn.softmax(sampler["scores"], where=topk_mask, initial=0)
        else:
            raise Exception(f"\"{self.prioritization}\" not a valid prioritization.")
        
        return w_s
    
    def staleness_weights(self, sampler: Sampler) -> chex.Array:
        """
        Returns staleness weights for each level.

        Args:
            sampler (Sampler): 

        Returns:
            chex.Array: shape (self.capacity)
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        staleness = sampler["episode_count"] - sampler["timestamps"]
        w_c = jnp.where(mask, staleness, 0)
        w_c = jax.lax.select(w_c.sum() > 0, w_c / w_c.sum(), mask / sampler["size"])
        return w_c
    
    def freshness_weights(self, sampler: Sampler) -> chex.Array:
        """
        Returns freshness weights for each level.

        Args:
            sampler (Sampler): 

        Returns:
            chex.Array: shape (self.capacity)
        """
        mask = jnp.arange(self.capacity) < sampler["size"]
        earliest_timestamp = jnp.where(mask, sampler["timestamps"], jnp.iinfo(jnp.int32).max).min()
        freshness = sampler["timestamps"] - earliest_timestamp
        w_f = jnp.where(mask, freshness, 0)
        w_f = jax.lax.select(w_f.sum() > 0, w_f / w_f.sum(), mask / sampler["size"])
        return w_f
    
    def flush(self, sampler: Sampler) -> Sampler:
        """
        Flushes this sampler, putting it back to its empty state. 
        This does update it in place.

        Args:
            sampler (Sampler): 

        Returns:
            Sampler:
        """
        sampler["size"] = 0
        sampler["scores"] = jnp.full(self.capacity, -jnp.inf, dtype=jnp.float32) 
        return sampler
    
    def _insert_new(self, sampler: Sampler, level: Level, score: float, level_extra: dict) -> Tuple[Sampler, int]:
        idx = self._get_next_idx(sampler)
        replace_cond = sampler["scores"][idx] < score
        
        def _replace():
            new_sampler = {
                **sampler,
                "levels": jax.tree_map(lambda x, y: x.at[idx].set(y), sampler["levels"], level),
                "scores": sampler["scores"].at[idx].set(score),
                "timestamps": sampler["timestamps"].at[idx].set(sampler["episode_count"] + 1),
                "size": jnp.minimum(sampler["size"] + 1, self.capacity),
            }
            if level_extra is not None:
                new_sampler["levels_extra"] = jax.tree_map(lambda x, y: x.at[idx].set(y), new_sampler["levels_extra"], level_extra)
            return new_sampler
            
        new_sampler = jax.lax.cond(replace_cond, _replace, lambda: sampler)
        new_sampler["episode_count"] += 1
        
        return new_sampler, jax.lax.select(replace_cond, idx, -1)
    
    def _proportion_filled(self, sampler: Sampler) -> float:
        return sampler["size"] / self.capacity
    
    def _get_next_idx(self, sampler: Sampler) -> int:
        return jax.lax.select(
            sampler["size"] < self.capacity,
            sampler["size"],
            self.level_weights(sampler).argmin()
        )
