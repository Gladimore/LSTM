local date = os.date("*t")
local time = os.clock()

local seed = date.sec * date.min * math.floor(time * 10^6)
math.randomseed(seed)

-- Utility functions
local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

local function tanh(x)
    return math.tanh(x)
end

local function sigmoid_derivative(x)
    local s = sigmoid(x)
    return s * (1 - s)
end

local function tanh_derivative(x)
    local t = tanh(x)
    return 1 - t * t
end

-- LSTM cell
local LSTMCell = {}
LSTMCell.__index = LSTMCell

function LSTMCell.new(input_size, hidden_size)
    local self = setmetatable({}, LSTMCell)
    self.input_size = input_size
    self.hidden_size = hidden_size

    -- Initialize weights and biases
    self.Wf = self:init_weights(hidden_size, input_size + hidden_size)
    self.Wi = self:init_weights(hidden_size, input_size + hidden_size)
    self.Wc = self:init_weights(hidden_size, input_size + hidden_size)
    self.Wo = self:init_weights(hidden_size, input_size + hidden_size)

    self.bf = self:init_bias(hidden_size)
    self.bi = self:init_bias(hidden_size)
    self.bc = self:init_bias(hidden_size)
    self.bo = self:init_bias(hidden_size)

    return self
end

function LSTMCell:init_weights(rows, cols)
    local weights = {}
    for i = 1, rows do
        weights[i] = {}
        for j = 1, cols do
            weights[i][j] = math.random() * 0.1 - 0.05
        end
    end
    return weights
end

function LSTMCell:init_bias(size)
    local bias = {}
    for i = 1, size do
        bias[i] = 0
    end
    return bias
end

function LSTMCell:forward(x, prev_h, prev_c)
    local combined = {}
    for i = 1, self.hidden_size + self.input_size do
        combined[i] = i <= self.input_size and x[i] or prev_h[i - self.input_size]
    end

    local f = self:apply_layer(self.Wf, self.bf, combined, sigmoid)
    local i = self:apply_layer(self.Wi, self.bi, combined, sigmoid)
    local c_tilde = self:apply_layer(self.Wc, self.bc, combined, tanh)
    local o = self:apply_layer(self.Wo, self.bo, combined, sigmoid)

    local c = {}
    for j = 1, self.hidden_size do
        c[j] = f[j] * prev_c[j] + i[j] * c_tilde[j]
    end

    local h = {}
    for j = 1, self.hidden_size do
        h[j] = o[j] * tanh(c[j])
    end

    -- Store intermediate values for backpropagation
    self.last_x = x
    self.last_prev_h = prev_h
    self.last_prev_c = prev_c
    self.last_f = f
    self.last_i = i
    self.last_c_tilde = c_tilde
    self.last_o = o
    self.last_c = c
    self.last_h = h

    return h, c
end

function LSTMCell:backward(dh, dc)
    local combined = {}
    for i = 1, self.hidden_size + self.input_size do
        combined[i] = i <= self.input_size and self.last_x[i] or self.last_prev_h[i - self.input_size]
    end

    local do_ = {}
    local df = {}
    local di = {}
    local dc_tilde = {}

    for j = 1, self.hidden_size do
        do_[j] = dh[j] * tanh(self.last_c[j])
        dc[j] = dc[j] + dh[j] * self.last_o[j] * (1 - tanh(self.last_c[j]) ^ 2)
        df[j] = dc[j] * self.last_prev_c[j]
        di[j] = dc[j] * self.last_c_tilde[j]
        dc_tilde[j] = dc[j] * self.last_i[j]
    end

    local dWf, dWi, dWc, dWo = {}, {}, {}, {}
    local dbf, dbi, dbc, dbo = {}, {}, {}, {}

    for i = 1, self.hidden_size do
        dbf[i], dWf[i] = self:backward_layer(self.Wf[i], df[i], combined, sigmoid_derivative)
        dbi[i], dWi[i] = self:backward_layer(self.Wi[i], di[i], combined, sigmoid_derivative)
        dbc[i], dWc[i] = self:backward_layer(self.Wc[i], dc_tilde[i], combined, tanh_derivative)
        dbo[i], dWo[i] = self:backward_layer(self.Wo[i], do_[i], combined, sigmoid_derivative)
    end

    local dx = {}
    local dprev_h = {}

    for i = 1, self.input_size do
        dx[i] = 0
        for j = 1, self.hidden_size do
            dx[i] = dx[i] + dWf[j][i] + dWi[j][i] + dWc[j][i] + dWo[j][i]
        end
    end

    for i = 1, self.hidden_size do
        dprev_h[i] = 0
        for j = 1, self.hidden_size do
            dprev_h[i] = dprev_h[i] + dWf[j][self.input_size + i] + dWi[j][self.input_size + i] +
            dWc[j][self.input_size + i] + dWo[j][self.input_size + i]
        end
    end

    local dprev_c = {}
    for i = 1, self.hidden_size do
        dprev_c[i] = dc[i] * self.last_f[i]
    end

    return { dWf = dWf, dWi = dWi, dWc = dWc, dWo = dWo, dbf = dbf, dbi = dbi, dbc = dbc, dbo = dbo }, dx, dprev_h,
        dprev_c
end

function LSTMCell:apply_layer(weights, bias, input, activation)
    local output = {}
    for i = 1, #weights do
        local sum = bias[i]
        for j = 1, #input do
            sum = sum + weights[i][j] * input[j]
        end
        output[i] = activation(sum)
    end
    return output
end

function LSTMCell:backward_layer(weights, grad_output, input, activation_derivative)
    local grad_bias = grad_output
    local grad_weights = {}

    for i = 1, #weights do
        grad_weights[i] = grad_output * activation_derivative(input[i])
    end

    return grad_bias, grad_weights
end

-- LSTM network
local LSTM = {}
LSTM.__index = LSTM

function LSTM.new(vocab_size, embedding_size, hidden_size, num_layers)
    local self = setmetatable({}, LSTM)
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    -- Initialize embedding
    self.embedding = self:init_embedding(vocab_size, embedding_size)

    -- Initialize LSTM layers
    self.layers = {}
    for i = 1, num_layers do
        local input_size = i == 1 and embedding_size or hidden_size
        self.layers[i] = LSTMCell.new(input_size, hidden_size)
    end

    -- Initialize output layer
    self.Wy = self:init_weights(vocab_size, hidden_size)
    self.by = self:init_bias(vocab_size)

    return self
end

function LSTM:init_embedding(vocab_size, embedding_size)
    local embedding = {}
    for i = 1, vocab_size do
        embedding[i] = {}
        for j = 1, embedding_size do
            embedding[i][j] = math.random() * 0.1 - 0.05
        end
    end
    return embedding
end

function LSTM:init_weights(rows, cols)
    local weights = {}
    for i = 1, rows do
        weights[i] = {}
        for j = 1, cols do
            weights[i][j] = math.random() * 0.1 - 0.05
        end
    end
    return weights
end

function LSTM:init_bias(size)
    local bias = {}
    for i = 1, size do
        bias[i] = 0
    end
    return bias
end

function LSTM:forward(input_sequence)
    local seq_length = #input_sequence
    local outputs = {}

    local h = {}
    local c = {}
    for i = 1, self.num_layers do
        h[i] = self:init_bias(self.hidden_size)
        c[i] = self:init_bias(self.hidden_size)
    end

    self.layer_outputs = {}

    for t = 1, seq_length do
        local x = self.embedding[input_sequence[t]]

        self.layer_outputs[t] = {}
        for i = 1, self.num_layers do
            h[i], c[i] = self.layers[i]:forward(x, h[i], c[i])
            x = h[i]
            self.layer_outputs[t][i] = { h = h[i], c = c[i] }
        end

        local output = self:apply_output_layer(h[self.num_layers])
        outputs[t] = output
    end

    return outputs
end

function LSTM:backward(input_sequence, target_sequence, learning_rate)
    local seq_length = #input_sequence
    local total_loss = 0

    -- Compute loss and gradients
    local dh = {}
    local dc = {}
    for i = 1, self.num_layers do
        dh[i] = self:init_bias(self.hidden_size)
        dc[i] = self:init_bias(self.hidden_size)
    end

    local dWy = self:init_weights(self.vocab_size, self.hidden_size)
    local dby = self:init_bias(self.vocab_size)

    for t = seq_length, 1, -1 do
        local target = target_sequence[t]
        local output = self:apply_output_layer(self.layer_outputs[t][self.num_layers].h)

        -- Compute cross-entropy loss
        local loss = -math.log(output[target])
        total_loss = total_loss + loss

        -- Compute gradient of output layer
        local doutput = {}
        for i = 1, self.vocab_size do
            doutput[i] = output[i]
        end
        doutput[target] = doutput[target] - 1

        -- Update output layer parameters
        for i = 1, self.vocab_size do
            for j = 1, self.hidden_size do
                dWy[i][j] = dWy[i][j] + doutput[i] * self.layer_outputs[t][self.num_layers].h[j]
            end
            dby[i] = dby[i] + doutput[i]
        end

        -- Backpropagate through time
        local dx = {}
        for i = 1, self.hidden_size do
            dx[i] = 0
            for j = 1, self.vocab_size do
                dx[i] = dx[i] + doutput[j] * self.Wy[j][i]
            end
        end

        for i = self.num_layers, 1, -1 do
            local layer_grads, layer_dx, layer_dh, layer_dc = self.layers[i]:backward(dx, dc[i])

            -- Update layer parameters
            self:update_parameters(self.layers[i].Wf, layer_grads.dWf, learning_rate)
            self:update_parameters(self.layers[i].Wi, layer_grads.dWi, learning_rate)
            self:update_parameters(self.layers[i].Wc, layer_grads.dWc, learning_rate)
            self:update_parameters(self.layers[i].Wo, layer_grads.dWo, learning_rate)

            self:update_parameters(self.layers[i].bf, layer_grads.dbf, learning_rate)
            self:update_parameters(self.layers[i].bi, layer_grads.dbi, learning_rate)
            self:update_parameters(self.layers[i].bc, layer_grads.dbc, learning_rate)
            self:update_parameters(self.layers[i].bo, layer_grads.dbo, learning_rate)

            dx = layer_dx
            dh[i] = layer_dh
            dc[i] = layer_dc
        end

        -- Update embedding
        local dembedding = dx
        for i = 1, self.embedding_size do
            self.embedding[input_sequence[t]][i] = self.embedding[input_sequence[t]][i] - learning_rate * dembedding[i]
        end
    end

    -- Update output layer parameters
    self:update_parameters(self.Wy, dWy, learning_rate)
    self:update_parameters(self.by, dby, learning_rate)

    return total_loss / seq_length
end

function LSTM:update_parameters(params, grads, learning_rate)
    if type(params) == "table" then
        for i = 1, #params do
            if type(params[i]) == "table" then
                for j = 1, #params[i] do
                    params[i][j] = params[i][j] - learning_rate * grads[i][j]
                end
            else
                params[i] = params[i] - learning_rate * grads[i]
            end
        end
    else
        params = params - learning_rate * grads
    end
end

function LSTM:apply_output_layer(h)
    local output = {}
    for i = 1, self.vocab_size do
        local sum = self.by[i]
        for j = 1, self.hidden_size do
            sum = sum + self.Wy[i][j] * h[j]
        end
        output[i] = math.exp(sum)
    end

    -- Softmax
    local total = 0
    for i = 1, self.vocab_size do
        total = total + output[i]
    end
    for i = 1, self.vocab_size do
        output[i] = output[i] / total
    end

    return output
end

function LSTM:generate(seed_sequence, num_chars)
    local generated = {}
    for i = 1, #seed_sequence do
        generated[i] = seed_sequence[i]
    end

    local h = {}
    local c = {}
    for i = 1, self.num_layers do
        h[i] = self:init_bias(self.hidden_size)
        c[i] = self:init_bias(self.hidden_size)
    end

    -- Process seed sequence
    for t = 1, #seed_sequence do
        local x = self.embedding[seed_sequence[t]]
        for i = 1, self.num_layers do
            h[i], c[i] = self.layers[i]:forward(x, h[i], c[i])
            x = h[i]
        end
    end

    -- Generate new characters
    for t = #seed_sequence + 1, num_chars do
        local x = self.embedding[generated[t - 1]]
        for i = 1, self.num_layers do
            h[i], c[i] = self.layers[i]:forward(x, h[i], c[i])
            x = h[i]
        end

        local output = self:apply_output_layer(h[self.num_layers])
        local next_char = self:sample(output)
        generated[t] = next_char
    end

    return generated
end

function LSTM:sample(probabilities)
    local r = math.random()
    local sum = 0
    for i = 1, self.vocab_size do
        sum = sum + probabilities[i]
        if r <= sum then
            return i
        end
    end
    return self.vocab_size
end

local json = require("dkjson")
-- Save function
function LSTM:save(filename)
    local model_data = {
        vocab_size = self.vocab_size,
        embedding_size = self.embedding_size,
        hidden_size = self.hidden_size,
        num_layers = self.num_layers,
        embedding = self.embedding,
        Wy = self.Wy,
        by = self.by,
        layers = {}
    }

    for _, layer in ipairs(self.layers) do
        table.insert(model_data.layers, {
            Wf = layer.Wf,
            Wi = layer.Wi,
            Wc = layer.Wc,
            Wo = layer.Wo,
            bf = layer.bf,
            bi = layer.bi,
            bc = layer.bc,
            bo = layer.bo
        })
    end

    -- Convert to JSON string
    local json_str = json.encode(model_data, { indent = true })

    -- Save to file
    local file = io.open(filename, "w")
    file:write(json_str)
    file:close()
end

-- Load function
function LSTM.load(filename)
    -- Read from file
    local file = io.open(filename, "r")
    local json_str = file:read("*a")
    file:close()

    -- Decode JSON string
    local model_data = json.decode(json_str)

    -- Create LSTM instance
    local self = LSTM.new(model_data.vocab_size, model_data.embedding_size, model_data.hidden_size, model_data.num_layers)

    -- Restore weights and biases
    self.embedding = model_data.embedding
    self.Wy = model_data.Wy
    self.by = model_data.by

    for i, layer_data in ipairs(model_data.layers) do
        self.layers[i].Wf = layer_data.Wf
        self.layers[i].Wi = layer_data.Wi
        self.layers[i].Wc = layer_data.Wc
        self.layers[i].Wo = layer_data.Wo
        self.layers[i].bf = layer_data.bf
        self.layers[i].bi = layer_data.bi
        self.layers[i].bc = layer_data.bc
        self.layers[i].bo = layer_data.bo
    end

    return self
end

return LSTM
