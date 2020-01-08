#!/usr/bin/env julia
progname=basename(@__FILE__)
doc = """ Kaggle Santa 2019

Usage:
  $(progname) <result> [--n= --epoch= --family= --assign= --dir= --freq=]
  $(progname) <result> [--resume= --epoch= --family= --assign= --dir= --freq=]

Args:
  <result>     semi-optimized assinged day result (*.csv) (w)

Options:
  --n=VAL           # of bees for ABC algorithm [default: 100]
  --epoch=VAL       # of iteration for ABC algorithm [default: 10000]
  --assign=FNAME    assigned day csv datafile
  --family=FNAME    family data csv [default: family_data.csv]
  --resume=FNAME    snapshot data (*.bson)
  --dir=PATH        directory path for snapshot [default: snapshot]
  --freq=VAL        reporting frequency [default: 10]

"""
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using DocOpt
args = docopt(doc)

using DataFrames
using Query
using CSVFiles
using DataStructures
using Distributions
using Random
using BSON: @save, @load
using ProgressMeter

include("./abc.jl")
using .ArtificialBeeColony

const N_FAMILIES = 5000
const N_DAYS = 100
const MAX_OCCUPANCY = 300
const MIN_OCCUPANCY = 125
const OCCUPANCY_PENALTY = 10^9
const PCOST_OFFSET = Int64[0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
const PCOST_GAIN = Int64[0, 0, 9, 9, 9, 18, 18, 36, 36, 36+199, 36+398]

function accounting_cost_func(Nd_curr, Nd_prev)
  (Nd_curr-125.0) / 400.0 * Nd_curr^(0.5 + abs(Nd_curr - Nd_prev) / 50.0) |> x->max(0, x)
end

function occupancy_penalty(dc::Vector{Int64})
  map(x->(x>MAX_OCCUPANCY || x<MIN_OCCUPANCY) ? OCCUPANCY_PENALTY : 0, dc) |> sum
end

function cost_func(m_assign::Array{Int64})
  # calculate the preference cost
  m_match_bin = (M_CHOICE .== m_assign)
  m_match10_bin = ones(eltype(m_assign), size(m_assign)) - sum(m_match_bin, dims=2)
  m_bin = hcat(m_match_bin, m_match10_bin)

  preference_cost = m_bin * PCOST_OFFSET + m_bin * PCOST_GAIN .* M_NPEOPLE |> sum

  # calculate daily occupancy cost (use soft constraints)
  daily_occupancy = zeros(eltype(m_assign), N_DAYS)
  for (idx, value) in enumerate(m_assign)
    daily_occupancy[value] += M_NPEOPLE[idx]
  end
  occupancy_penalty = map(x->(x>MAX_OCCUPANCY || x<MIN_OCCUPANCY) ? OCCUPANCY_PENALTY : 0, daily_occupancy) |> sum

  # calulate the accounting cost
  accounting_cost = 0.0
  Nd_prev = daily_occupancy[N_DAYS]

  for day in N_DAYS:-1:1
    Nd_curr = daily_occupancy[day]
    accounting_cost += accounting_cost_func(Nd_curr, Nd_prev)
    Nd_prev = Nd_curr
  end

  # sum all the costs and the penalty
  penalty = preference_cost + accounting_cost + occupancy_penalty

  return penalty
end

function init()
  rand(1:N_DAYS, N_FAMILIES)
end

# target function
function target(x::Vector{Int64})
  cost_func(x)
end

function save_assigned_day(fname::String, data::Vector)
  d = hcat(collect(0:N_FAMILIES-1), data) |> DataFrame
  rename!(d, [:family_id, :assigned_day])
  d |> save(fname)
end

function main(args)
  # load assign data
  if isnothing(args["--assign"])
    m_assign = rand(1:N_DAYS, N_FAMILIES)
  else
    df_assign = load(args["--assign"]) |> DataFrame
    m_assign = df_assign |> @select(:assigned_day) |> DataFrame |> Matrix
  end

  # load family_data
  df_family = load(args["--family"]) |> DataFrame
  global M_CHOICE = df_family |> @select(-:n_people,-:family_id) |> DataFrame |> Matrix
  global M_NPEOPLE = df_family |> @select(:n_people) |> DataFrame |> Matrix

  # create snapshot dir
  snapshot_dir = args["--dir"]
  !isdir(snapshot_dir) && mkdir(args["--dir"])

  # optimize the day assignment by using Artificial Bee Colony algorithm
  if isnothing(args["--resume"])
    N = parse(Int64, args["--n"])
    abc = ABC(N, init, target)
  else
    @load args["--resume"] abc
    N = length(abc)
  end
  epoch = parse(Int64, args["--epoch"])

  # set pre-obtained day assignment
  best = vec(m_assign)
  init_manually!(abc, best)

  # report frequency
  F1 = parse(Int64, args["--freq"])
  p = Progress(round(Int64, epoch/F1), barlen=10)
  
  # handle interrupt
  ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

  # cost history buffer for reporting
  costs = CircularBuffer{Tuple{Int64,Float64}}(10)

  try
    for i in 1:epoch
      best = search_epoch1!(abc)

      # progress
      if i % F1 == 0
        save_assigned_day(joinpath(snapshot_dir, "abc_$i.csv"), best)
        push!(costs, (i,cost_func(best)))
        next!(p, showvalues=[(:iter,i),
                             [(Symbol("iter_$(costs[j][1])"), costs[j][2]) for j in length(costs):-1:1]...])
      end
    end
  catch
    println("\ninterrupt..")
  end

  @save joinpath(snapshot_dir, splitext(args["<result>"])[1]*".bson") abc
  save_assigned_day(args["<result>"], best)
  println(cost_func(best))
end

main(args)

