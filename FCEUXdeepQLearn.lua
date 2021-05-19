--Welcome to FCEUXDeepQlearn! Like the name suggests, this script attempts to implement deep Q learning to play a game on the NES emulator, FCEUX. Please set the variables anf functions below, and enjoy! This code should load a ROM, and play a movie(You have to specify precisely which it should use). When the movie finishes playing, the AI will take over and try to maximize the output of the utility function(which has to be defined depending on the game you want it to play) for a certain number of frames. To stop it, press start and select together, and it will create a movie with the inputs it has made.

-------------------------Variables that you should set---------------------------

screenX = 16
screenY = 16   --Resolutions of downscaled screen, the input of the neural network.
howmanyinputs = 4  --The number of buttons that are allowed to press.(Exactly which can be changed later, in line 299.)
netstructure = {160, 80} --The hidden layers of the neural network. each entry in the table corresponds to the number of nodes in each layer. The length of the table can be changed, by the way. It isn't limited to 2. This is just an example.
ROMlocation = "" --Which ROM should it play?
primaryMovieLocation = "" --Which movie should it play before letting the AI play? This is commonly used to get past the title screen, or to start the AI in a particular section.
secondaryMovieLocation = "" --What movie should it output?
time = 600 --How long is it allowed to play?
NetworkLearnRate = 5 * 10^-15 --What is the learning rate of the network?(I advise you to experiment with this number. You can uncomment line 206 to help with this,
trainiterationsperstep = 10 --How many training cycles should it do every frame? lower is faster, but it learns more slowly.
discount = 0.5 --How much should it care about the future? This should be between 0 and 1. If the number is higher, it cares about the future more and more.
RNG = 0.5 --How much should it try random inputs? The AI has to stumble across a way to get a higher score, and occasionally making it do random things helps with that. This number should also be between - and 1. The higher this number is, the more random the AI's behavior becomes.
RNGfalloff = 0.999 --How slowly should the randomness decrease over time? This number should also be between 0 and 1, and the higher it is, the slower the randomness is decreased.
main = savestate.object(5) --Make sure these savestate slots are not used by you! This script will overwrite them.
buffer = savestate.object(6)
speed = "turbo"  --What speedmode should the game run on? Choose between "normal", "turbo", and "maximum".

-----------------------------------Please define these functions as well.--------------------------

function utility() --This gives the value the code tried to maximize. Commonly a score, or how far to the right a character is, etc. Please define this function depending on the game you're playing.
  return 0
end
function dead() --If this function return true, it will immediately reset the game and start over. This is commonly used to detect a death and reset the game when it happens. Also, the reward will be zero in order to make the AI hopefully avoid it. Again, you have to define this function properly.
  return false
end

--------------------------------------The code will take care of the rest :)------------------------------
function biggest(list)
  local big = {1}
  for x = 2, #list do
    if list[x] > list[big[1]] then
      big = {x}
    elseif list[x] == list[big[1]] and big[1] ~= x then
      table.insert(big,x)
    end
  end
  return big[math.random(1,#big)]
end

function sigmoid(x)
  return 1 / (1 + math.exp(x))
end

function relu(x)
  if x > 0 then
    return x
  else
    return x / 1024
  end
end

function activation(x)
  return relu(x)
end

function derivative(x)
  return (activation(x + 10^(-8)) - activation(x)) * 10^8
end

function clone(x)
  if type(x) == "table" then
    local a = {}
    for q = 1, #x do
      table.insert(a, clone(x[q]))
    end
    return a
  else
    return x
  end
end

function network(nodes)
  local s = {}
  s.nodes = {}
  s.raw = {}
  s.weights = {}
  s.biases = {}
  s.cost = 0
  local a = {}
  for x = 1, nodes[1] do
    table.insert(a, 0)
  end
  table.insert(s.nodes, a)
  table.insert(s.raw, a)
  for x = 1, #nodes - 1 do
    table.insert(s.nodes, {})
    table.insert(s.raw, {})
    table.insert(s.weights, {})
    table.insert(s.biases, {})
    for y = 1, nodes[x + 1] do
      table.insert(s.nodes[x + 1], 0)
      table.insert(s.raw[x + 1], 0)
      table.insert(s.weights[x], {})
      table.insert(s.biases[x], (2 * (math.random() - 0.5)) ^ 3)
      for z = 1, nodes[x] do
        table.insert(s.weights[x][y], (2 * (math.random() - 0.5)) ^ 3)
      end
    end
  end
  return s
end

function predict(se, input)
  local s = se
  s.nodes[1] = input
  s.raw[1] = input
  local a = {}
  local c = {}
  local b = 0
  for x = 1, #s.weights do
    a = {}
    c = {}
    for y = 1, #s.weights[x] do
      b = s.biases[x][y]
      for z = 1, #s.weights[x][y] do
        b = b + s.weights[x][y][z] * s.nodes[x][z]
      end
      table.insert(a, activation(b))
      table.insert(c, b)
    end
    s.nodes[x + 1] = a
    s.raw[x + 1] = c
  end
  return s
end

function output(s)
  return s.nodes[#s.nodes]
end

function cost(se, input, outputt)
  local s = predict(se, input)
  local a = output(s)
  local b = 0
  for x = 1, #a do
    b = b + (a[x] - outputt[x])^2
  end
  return b
end

function backprop(se, input, outputt)
  local s = predict(se, input)
  local w = clone(s.weights)
  local b = clone(s.biases)
  local expectedoutput = outputt
  local differences = {}
  local x = 0
  local a = 0
  for p = 1, #s.weights do
    x = #s.nodes - p
    differences = {}
    for y = 1, #s.nodes[x + 1] do
      table.insert(differences, s.nodes[x + 1][y] - expectedoutput[y])
      b[x][y] = 2 * differences[y] * derivative(s.raw[x + 1][y])
      for z = 1, #s.nodes[x] do
        w[x][y][z] = s.nodes[x][z] * 2 * differences[y] * derivative(s.raw[x + 1][y])
      end
    end
    expectedoutput = {}
    for y = 1, #s.nodes[x] do
      a = 0
      for z = 1, #s.nodes[x + 1] do
        a = a + s.weights[x][z][y] * 2 * differences[z] * derivative(s.raw[x + 1][z])
      end
      table.insert(expectedoutput, ((a / #s.nodes[x + 1]) / -2) + s.nodes[x][y])
    end
  end
  return {w, b}
end


function train(se, inputs, outputs, LearnRate, iterations)
  local s = se
  local avgw = clone(s.weights)
  local avgb = clone(s.biases)
  local avgCost = 0
  local c = {}
  local total = 0
  for q = 1, iterations do
    avgCost = 0
    for r = 1, #inputs do
      c = backprop(s, inputs[r], outputs[r])
      avgCost = avgCost + cost(s, inputs[r], outputs[r]) / #inputs
      for x = 1, #s.weights do
        for y = 1, #s.weights[x] do
          avgb[x][y] = avgb[x][y] + c[2][x][y]
          for z = 1, #s.weights[x][y] do
            avgw[x][y][z] = avgw[x][y][z] + c[1][x][y][z] / #inputs
          end
        end
      end
    end
    for x = 1, #s.weights do
      for y = 1, #s.weights[x] do
        s.biases[x][y] = s.biases[x][y] - avgb[x][y] * LearnRate * math.sqrt(avgCost)
        for z = 1, #s.weights[x][y] do
          s.weights[x][y][z] = s.weights[x][y][z] - avgw[x][y][z] * LearnRate * math.sqrt(avgCost)
        end
      end
    end
    s.cost = avgCost
    --print(avgCost) --This is the line to show the cost change every training cycle, this can be used for tuning the learning rate. Make sure it decreases, but at a reasonable pace.
  end
  return s
end

function tobinary(n, digits)
  local number = n
  local a = {}
  for q = 1, digits do
    x = digits - q
    if number > 2^x then
      number = number - 2^x
      table.insert(a, true)
    else
      table.insert(a, false)
    end
  end
  return a
end

function getscreen(screenX, screenY)
  local screen = {}
  local value = 0
  local r = 0
  local g = 0
  local b = 0
  local pallete = 0
  for x = 0, screenX - 1 do
    for y = 0, screenY - 1 do
      gui.pixel(x * math.floor(255 / screenX), y * math.floor(239 / screenY), "cyan")
      value = 0
      for z = 0, math.floor(255 / screenX) - 1 do
        for w = 0, math.floor(239 / screenY) - 1 do
          r, g, b, pallete = emu.getscreenpixel(x * math.floor(255 / screenX) + z,y * math.floor(239 / screenY) + w, true)
          value = value + r + g + b
        end
      end
      table.insert(screen, value / math.floor((256 * 240) / (screenX * screenY)))
    end
  end
  return screen
end

networkdimensions = {}
table.insert(networkdimensions, screenX * screenY)
for x = 1, #netstructure do
  table.insert(networkdimensions, netstructure[x])
end
table.insert(networkdimensions, 2^howmanyinputs)

q = network(networkdimensions)

---------------load ROM, play until finished--------
dummyarray = {}
emu.loadrom(ROMlocation)
emu.speedmode("maximum")
movie.play(primaryMovieLocation, true)
while movie.mode() ~= "finished" do
  emu.frameadvance()
  table.insert(dummyarray, joypad.get(1))
  end
movie.stop()
movie.record(secondaryMovieLocation)
movie.rerecordcounting(true)
for x = 1, #dummyarray do
  joypad.set(1, dummyarray[x])
  emu.frameadvance()
end
savestate.save(main)
emu.speedmode(speed)

----------------play--------------
count = 0
bestiteverdid = 0
qs = {}
inputs = {}
screendata = {}
reward = 0
futurereward = 0
futureqs = {}
bestfinalscore = 0
yourinputs = {}
terminated = false
while true do
  count = count + 1
  savestate.load(main)
  for t = 1, time do
    screendata = getscreen(screenX, screenY)
    q = predict(q, screendata)
    qs = output(q)
    thebestq = biggest(qs) - 1
    if RNG > math.random() then
      thebestq = math.random(0, 2^howmanyinputs - 1)
    end
    inputs = tobinary(thebestq, howmanyinputs)
    joypad.set(1, {A=inputs[1], right=inputs[2], B=inputs[3], left=inputs[4]}) --Change possible inputs here.
    emu.frameadvance()
    reward = utility()
    if reward > bestiteverdid then
      bestiteverdid = reward
    end
    q = predict(q, getscreen(screenX, screenY))
    futureqs = output(q)
    futurereward = futureqs[biggest(futureqs)]
    if dead() then
      qs[thebestq] = 0
      q = train(q, {screendata}, {qs}, NetworkLearnRate, trainiterationsperstep)
      break
    end
    qs[thebestq] = reward + futurereward * discount
    q = train(q, {screendata}, {qs}, NetworkLearnRate, trainiterationsperstep)
    gui.text(5, 10, "Episode: "..count.." Score: "..utility().."\nBest score ever: "..bestiteverdid.."\nCost: "..q.cost.."\nimprovement: "..q.cost - cost(q, screendata, qs).."\nRandomness: "..RNG.."\n"..time - t.." frames until reset")
    RNG = RNG * RNGfalloff
    yourinputs = joypad.read(1)
    if yourinputs.select and yourinputs.select then
      terminated = true
      break
    end
  end
  if terminated then
    break
  end
  if bestfinalscore <= utility() then
    bestfinalscore = utility()
    print("New record!")
    savestate.save(buffer)
  end
end
savestate.load(buffer)
movie.stop()
