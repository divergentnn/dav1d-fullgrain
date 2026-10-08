-- Live A/B of dav1d-fullgrain's synthesis modes inside mpv.
--
-- The decoder re-reads the file named by DAV1D_GRAIN_MODE_FILE every 100 ms;
-- this script writes it. contrib/mpv/mpv-fullgrain sets everything up; with
-- the fork installed system-wide, export DAV1D_GRAIN_MODE_FILE (any writable
-- path) before starting mpv and load this script (--script=... or copy it to
-- ~/.config/mpv/scripts/).
--
--   g        cycle standard -> dual -> full -> multi (shown on screen)
--   b        blind toggle between standard and full: shows only "A" or "B";
--            which is which is decided at random when mpv starts
--   B        reveal the blind mapping
--
-- When paused, the current frame is re-decoded so the change is visible
-- immediately; while playing it takes effect within a few frames.

local path = os.getenv("DAV1D_GRAIN_MODE_FILE")
local modes = { "standard", "dual", "full", "multi" }
local idx = 1

math.randomseed(os.time() + math.floor(mp.get_time() * 1000))
local blind = { "standard", "full" }
if math.random() < 0.5 then blind = { "full", "standard" } end
local blind_cur = 0

local function read_mode()
    if not path then return nil end
    local f = io.open(path, "r")
    if not f then return nil end
    local m = f:read("*l")
    f:close()
    return m
end

local function write_mode(m)
    if not path then
        mp.osd_message("grain-mode: DAV1D_GRAIN_MODE_FILE is not set (start mpv via mpv-fullgrain or export DAV1D_GRAIN_MODE_FILE)", 4)
        return false
    end
    local f = io.open(path, "w")
    if not f then return false end
    f:write(m .. "\n")
    f:close()
    if mp.get_property_bool("pause") then
        -- re-decode the frame on screen (the fork polls the file every 100 ms)
        mp.add_timeout(0.15, function() mp.commandv("seek", "0", "relative+exact") end)
    end
    return true
end

local cur = read_mode()
for i, m in ipairs(modes) do if m == cur then idx = i end end

mp.add_key_binding("g", "grain-cycle", function()
    idx = idx % #modes + 1
    if write_mode(modes[idx]) then mp.osd_message("grain: " .. modes[idx], 2) end
end)

mp.add_key_binding("b", "grain-blind", function()
    blind_cur = blind_cur % 2 + 1
    if write_mode(blind[blind_cur]) then
        mp.osd_message("grain: " .. (blind_cur == 1 and "A" or "B"), 2)
    end
end)

mp.add_key_binding("B", "grain-reveal", function()
    mp.osd_message("A = " .. blind[1] .. ", B = " .. blind[2], 4)
end)
