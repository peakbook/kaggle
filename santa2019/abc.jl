module ArtificialBeeColony

export ABC
export init_manually!, search_epoch1!
export get_best_data

import Base: isless, length

mutable struct Bee
  data :: Vector{Int64}
  fitness :: Float64
  count :: Int64
end

isless(a::Bee, b::Bee) = a.fitness < b.fitness

function Bee(data::Vector)::Bee
  return Bee(data, zero(Float64), zero(Int64))
end

const Bees = Vector{Bee}

function Base.copyto!(t::Bee, s::Bee)
  copyto!(t.data, s.data)
  t.fitness = s.fitness
  t.count = s.count
  return t
end

function Base.copy(s::Bee)::Bee
  return Bee(copy(s.data), s.fitness, s.count)
end

struct ABC
  bees :: Bees
  best :: Bee
  init :: Function
  cost :: Function
  function ABC(N::Integer, init::Function, cost::Function)::ABC
    bees = Bee[Bee(init()) for i=1:N]
    update_fitness!.(bees, cost)
    return new(bees, minimum(bees), init, cost)
  end
end

function length(a::ABC)
  return length(a.bees)
end

function update_fitness!(bee::Bee, g::Function)
  bee.fitness = fitness(g(bee.data))
  return
end

function update_bee!(bee::Bee, beem::Bee, g::Function)
  dim = length(bee.data)
  pos = rand(1:dim)

  m = bee.data[pos]
  bee.data[pos] = beem.data[pos]
  fitnew = fitness(g(bee.data))

  if fitnew > bee.fitness
    bee.fitness = fitnew
    bee.count = 0
  else
    bee.data[pos] = m
    bee.count += 1
  end
  return
end

function update_employed!(bees::Bees, g::Function)
  N = length(bees)
  update_bee!.(bees, bees[rand(1:N, N)], g)
  return
end

function roulette_select(bees::Bees)::Bee
  sf = sum(x->x.fitness, bees)
  r = rand()
  rs = zero(r)

  for bee in bees
    rs += bee.fitness/sf
    if r<=rs
      return bee
    end
  end
  return bees[end]
end

function update_outlook!(bees::Bees, g::Function, No::Integer)
  N = length(bees)
  #  update_bee!.(Bee[roulette_select(bees) for i in 1:No], bees[rand(1:N,No)], g)
  for i=1:No
    bee = roulette_select(bees)
    update_bee!(bee, bees[rand(1:N)], g)
  end
  return
end

function update_scout!(bees::Bees, g::Function, init::Function, limit::Integer)
  bees_scout = filter(x->x.count >= limit, bees)
  for bee in bees_scout
    bee.data = init()
    bee.count = 0
  end
  update_fitness!.(bees_scout, g)
  return
end

function update_best!(bees::Bees, bee_best::Bee)
  bee_cand = maximum(bees)
  if bee_best.fitness < bee_cand.fitness
    copyto!(bee_best, bee_cand)
  end
  return
end

function init_manually!(abc::ABC, data::Vector; pos::Integer=1)
  abc.bees[pos] = Bee(data)
  update_fitness!(abc.bees[pos], abc.cost)
  return
end

function search_epoch1!(abc::ABC,
                        No = length(abc.bees),
                        Ne = length(abc.bees),
                        limit=round(Integer,0.1*(No+Ne)*length(abc.bees[1].data)))::Vector
  update_employed!(abc.bees, abc.cost)
  update_outlook!(abc.bees, abc.cost, No)
  update_best!(abc.bees, abc.best)
  update_scout!(abc.bees, abc.cost, abc.init, limit)

  return abc.best.data
end

function get_best_data(abc::ABC)::Vector
  return abc.best.data
end

function fitness(val::T) where T<:AbstractFloat
  if val >= zero(T)
    return one(T)/(one(T)+val)
  else
    return one(T)+abs(val)
  end
end

end
