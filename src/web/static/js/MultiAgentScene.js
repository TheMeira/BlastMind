class MultiAgentScene extends Phaser.Scene {
    constructor() {
        super({ key: 'MultiAgentScene' });
    }

    init(data) {
        this.agentIds = (data && data.agents) || ['random'];
        this.speed    = (data && data.speed)  || 1.0;
        this.seed     = (data && data.seed)   || Math.floor(Math.random() * 999999);
        this.agents   = [];
        this.allDone  = false;
    }

    preload() {
        if (!this.cache.audio.exists('click')) {
            this.load.audio('click', '/static/audio/click.mp3');
        }
        if (!this.cache.audio.exists('line-clear')) {
            this.load.audio('line-clear', '/static/audio/line-clear.mp3');
        }
        if (!this.cache.audio.exists('game-over')) {
            this.load.audio('game-over', '/static/audio/game-over.mp3');
        }
        if (!this.cache.audio.exists('place')) {
            this.load.audio('place', '/static/audio/place.mp3');
        }
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');
        const W = this.scale.width;
        const H = this.scale.height;
        const N = this.agentIds.length;

        this.spawnBackground(W, H);

        const positions = this.computeLayout(W, H);

        this.agents = this.agentIds.map((id, i) => {
            const roster = (typeof AGENT_ROSTER !== 'undefined')
                ? AGENT_ROSTER.find(a => a.id === id)
                : null;
            const label = roster ? roster.label : id.charAt(0).toUpperCase() + id.slice(1);
            return {
                id,
                label,
                ws: null,
                state: null,
                prevBoard: null,
                colorGrid: Array.from({ length: BOARD_N }, () => Array(BOARD_N).fill(null)),
                gfx: this.add.graphics(),
                nameTxt: null,
                scoreTxt: null,
                finished: false,
                finalScore: 0,
                stats: { piecesPlaced: 0, linesCleared: 0, bestCombo: 0 },
                statTxts: null,
                trayGfx: this.add.graphics(),
                traySlots: null,
                lastPieceCount: -1,
                ...positions[i],
            };
        });

        const nameFS  = Math.floor(H * (N === 1 ? 0.026 : 0.019));
        const scoreFS = Math.floor(H * (N === 1 ? 0.022 : 0.016));

        this.agents.forEach(agent => {
            this.drawEmptyBoard(agent);

            agent.nameTxt = this.add.text(agent.labelX, agent.labelY - Math.floor(nameFS * 0.6), agent.label, {
                fontFamily: 'Orbitron, Arial',
                fontSize: nameFS + 'px',
                fontStyle: 'bold',
                color: '#00d4ff',
            }).setOrigin(0.5, 0.5);

            agent.scoreTxt = this.add.text(agent.labelX, agent.labelY + Math.floor(scoreFS * 0.7), '0', {
                fontFamily: 'Orbitron, Arial',
                fontSize: scoreFS + 'px',
                color: '#c8e8ff',
            }).setOrigin(0.5, 0.5);
        });

        if (N === 1) {
            this.buildStatsPanel(W, H, this.agents[0]);
        }

        this._lastPlaceSFX = 0;
        this._lastClearSFX = 0;

        this.buildSpeedControls(W, H);
        this.buildMenuButton(H);

        this.agents.forEach(agent => this.connectAgent(agent));

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', this._shutdown, this);
    }

    _shutdown() {
        this.scale.off('resize', this._onResize, this);
        this.agents.forEach(a => { if (a.ws) { a.ws.onmessage = null; a.ws.close(); } });
    }

    computeLayout(W, H) {
        const N = this.agentIds.length;
        const topPad = Math.floor(H * 0.09);
        const botPad = Math.floor(H * 0.04);
        const availH = H - topPad - botPad;

        if (N === 1) {
            const trayH = Math.floor(availH * 0.12);
            const boardAvailH = availH - trayH;
            const cellByH = Math.floor(boardAvailH / BOARD_N);
            const cellByW = Math.floor(W * 0.50 / BOARD_N);
            const CS = Math.min(100, cellByH, cellByW);
            const BP = BOARD_N * CS;
            const BX = Math.floor((W - BP) / 2);
            const BY = topPad + Math.floor((boardAvailH - BP) / 2);
            const labelX = Math.floor(W / 2);
            const labelY = Math.floor((topPad + BY) / 2);
            return [{ bx: BX, by: BY, cs: CS, bp: BP, labelX, labelY, trayY: BY + BP + Math.floor(H * 0.012), trayH }];
        }

        let cols, rows;
        if (N <= 3) { cols = N; rows = 1; }
        else if (N === 4) { cols = 2; rows = 2; }
        else { cols = 3; rows = 2; }

        const hPad    = Math.floor(W * 0.020);
        const vPad    = Math.floor(H * 0.030);
        const labelH  = Math.floor(H * 0.065);
        const cellW   = Math.floor((W - hPad * (cols + 1)) / cols);
        const cellH   = Math.floor((availH - vPad * (rows - 1)) / rows);
        const trayH   = Math.floor(cellH * 0.15);
        const boardAreaW = Math.floor(cellW * 0.88);
        const boardAreaH = cellH - labelH - trayH;

        const CS = Math.min(56, Math.floor(boardAreaW / BOARD_N), Math.floor(boardAreaH / BOARD_N));
        const BP = BOARD_N * CS;

        const positions = [];

        if (N === 5) {
            for (let i = 0; i < 3; i++) {
                const cellX = hPad + i * (cellW + hPad);
                const cellY = topPad;
                positions.push(this._cellPos(cellX, cellY, cellW, labelH, BP, CS, trayH));
            }
            const row2W      = 2 * cellW + hPad;
            const row2StartX = Math.floor((W - row2W) / 2);
            for (let i = 0; i < 2; i++) {
                const cellX = row2StartX + i * (cellW + hPad);
                const cellY = topPad + cellH + vPad;
                positions.push(this._cellPos(cellX, cellY, cellW, labelH, BP, CS, trayH));
            }
        } else {
            for (let i = 0; i < N; i++) {
                const row   = Math.floor(i / cols);
                const col   = i % cols;
                const cellX = hPad + col * (cellW + hPad);
                const cellY = topPad + row * (cellH + vPad);
                positions.push(this._cellPos(cellX, cellY, cellW, labelH, BP, CS, trayH));
            }
        }

        return positions;
    }

    _cellPos(cellX, cellY, cellW, labelH, BP, CS, trayH) {
        const by = Math.floor(cellY + labelH);
        return {
            bx:     Math.floor(cellX + (cellW - BP) / 2),
            by,
            cs:     CS,
            bp:     BP,
            labelX: Math.floor(cellX + cellW / 2),
            labelY: Math.floor(cellY + labelH / 2),
            trayY:  by + BP + 6,
            trayH,
        };
    }

    connectAgent(agent) {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${proto}://${location.host}/ws/ai`);
        agent.ws = ws;

        ws.onopen = () => {
            ws.send(JSON.stringify({ speed: this.speed, agent: agent.id, seed: this.seed }));
        };

        ws.onmessage = (evt) => {
            if (agent.finished) return;
            const newState = JSON.parse(evt.data);
            const lastPieceId = newState.last_piece_id || null;

            if (lastPieceId && agent.prevBoard) {
                const now = Date.now();
                if (now - this._lastPlaceSFX > 120) {
                    try { this.sound.play('place', { volume: getSFXVolume() }); } catch (e) {}
                    this._lastPlaceSFX = now;
                }
                this.updateColorGrid(agent, agent.prevBoard, newState.board, lastPieceId);
                this.updateStats(agent, agent.prevBoard, newState);
            }

            agent.state = newState;
            agent.prevBoard = newState.board.map(row => [...row]);

            this.updateTray(agent, newState, lastPieceId);

            agent.scoreTxt.setText(String(newState.score));
            this.renderAgentBoard(agent);

            if (this.agents.length === 1 && agent.statTxts) {
                this.refreshStatsTxts(agent);
            }

            if (newState.game_over) {
                agent.finished   = true;
                agent.finalScore = newState.score;
                try { this.sound.play('game-over', { volume: getSFXVolume() }); } catch (e) {}
                this.renderAgentBoard(agent);
                this.checkAllDone();
            }
        };

        ws.onerror = () => {
            agent.finished = true;
            this.checkAllDone();
        };
    }

    updateColorGrid(agent, prevBoard, newBoard, pieceId) {
        const color = (typeof PIECE_COLORS !== 'undefined' && PIECE_COLORS[pieceId]) || 0x448aff;
        for (let r = 0; r < BOARD_N; r++) {
            for (let c = 0; c < BOARD_N; c++) {
                if (newBoard[r][c] === 0) {
                    agent.colorGrid[r][c] = null;
                } else if (newBoard[r][c] === 1 && prevBoard[r][c] === 0) {
                    agent.colorGrid[r][c] = color;
                }
            }
        }
    }

    updateStats(agent, prevBoard, newState) {
        if (newState.last_piece_id) agent.stats.piecesPlaced++;

        let lines = 0;
        if (typeof newState.lines_cleared === 'number') {
            lines = newState.lines_cleared - agent.stats.linesCleared;
            agent.stats.linesCleared = newState.lines_cleared;
        }

        if (newState.combo_count > agent.stats.bestCombo) {
            agent.stats.bestCombo = newState.combo_count;
        }

        if (lines > 0) {
            const now = Date.now();
            if (now - this._lastClearSFX > 120) {
                try { this.sound.play('line-clear', { volume: getSFXVolume() }); } catch (e) {}
                this._lastClearSFX = now;
            }
        }
    }

    updateTray(agent, newState, lastPieceId) {
        const pieces = newState.pieces || [];

        if (!agent.traySlots || pieces.length > agent.lastPieceCount) {
            agent.traySlots = pieces.map(p => ({ id: p.id, grid: p.grid, used: false }));
        } else if (pieces.length < agent.lastPieceCount && lastPieceId) {
            const slot = agent.traySlots.find(s => !s.used && s.id === lastPieceId);
            if (slot) slot.used = true;
        }

        agent.lastPieceCount = pieces.length;
        this.renderTray(agent);
    }

    renderTray(agent) {
        const gfx = agent.trayGfx;
        gfx.clear();
        if (!agent.traySlots) return;

        const { bx, bp, trayY, trayH } = agent;
        const slotW = Math.floor(bp / 3);
        const mc = Math.max(3, Math.floor(Math.min(slotW, trayH) / 5.6));

        gfx.fillStyle(0x0a0a1e, 0.85);
        gfx.fillRoundedRect(bx - 4, trayY, bp + 8, trayH, 6);

        agent.traySlots.forEach((slot, i) => {
            const grid = slot.grid;
            const ph = grid.length;
            const pw = grid[0].length;
            const cx = bx + i * slotW + slotW / 2;
            const cy = trayY + trayH / 2;
            const x0 = cx - (pw * mc) / 2;
            const y0 = cy - (ph * mc) / 2;
            const color = (typeof PIECE_COLORS !== 'undefined' && PIECE_COLORS[slot.id]) || 0x448aff;

            gfx.fillStyle(color, slot.used ? 0.12 : 0.9);
            for (let r = 0; r < ph; r++) {
                for (let c = 0; c < pw; c++) {
                    if (grid[r][c] === 1) {
                        gfx.fillRoundedRect(x0 + c * mc + 1, y0 + r * mc + 1, mc - 2, mc - 2, 2);
                    }
                }
            }
        });
    }

    spawnBackground(W, H) {
        const colors = [0x448aff, 0xff6d00, 0xffea00, 0xff1744, 0x00e676, 0xe040fb, 0x7c4dff, 0x00e5ff];
        this.bgBlocks = [];

        const count = Math.floor((W * H) / 30000);

        for (let i = 0; i < count; i++) {
            const size = Phaser.Math.Between(36, Math.floor(W * 0.08));
            const x = Phaser.Math.Between(-80, W + 80);
            const y = Phaser.Math.Between(-80, H + 80);
            const color = colors[i % colors.length];
            const alpha = Phaser.Math.FloatBetween(0.05, 0.14);

            const rect = this.add.rectangle(x, y, size, size, color, alpha);
            rect.rotation = Phaser.Math.FloatBetween(0, Math.PI * 2);

            this.bgBlocks.push({
                rect,
                vx: Phaser.Math.FloatBetween(-0.5, 0.5),
                vy: Phaser.Math.FloatBetween(-0.9, -0.2),
                rs: Phaser.Math.FloatBetween(-0.004, 0.004),
                W, H,
            });
        }
    }

    update() {
        if (!this.bgBlocks) return;
        this.bgBlocks.forEach(b => {
            b.rect.x += b.vx;
            b.rect.y += b.vy;
            b.rect.rotation += b.rs;
            if (b.rect.y < -200) {
                b.rect.y = b.H + 100;
                b.rect.x = Phaser.Math.Between(-80, b.W + 80);
            }
            if (b.rect.x < -200) b.rect.x = b.W + 100;
            if (b.rect.x > b.W + 200) b.rect.x = -100;
        });
    }

    drawEmptyBoard(agent) {
        const { gfx, bx, by, cs, bp } = agent;
        const pad = Math.max(2, Math.floor(cs * 0.055));
        const r   = Math.max(3, Math.floor(cs * 0.08));
        gfx.clear();
        gfx.fillStyle(0x0a0a1e);
        gfx.fillRoundedRect(bx - 4, by - 4, bp + 8, bp + 8, 6);
        for (let row = 0; row < BOARD_N; row++) {
            for (let col = 0; col < BOARD_N; col++) {
                gfx.fillStyle(0x181830);
                gfx.fillRoundedRect(bx + col * cs + pad, by + row * cs + pad, cs - pad * 2, cs - pad * 2, r);
            }
        }
    }

    renderAgentBoard(agent) {
        if (!agent.state) return;
        const { gfx, bx, by, cs, bp, colorGrid, finished } = agent;
        const board = agent.state.board;
        const pad   = Math.max(2, Math.floor(cs * 0.055));
        const r     = Math.max(3, Math.floor(cs * 0.08));

        gfx.clear();
        gfx.fillStyle(0x0a0a1e);
        gfx.fillRoundedRect(bx - 4, by - 4, bp + 8, bp + 8, 6);

        for (let row = 0; row < BOARD_N; row++) {
            for (let col = 0; col < BOARD_N; col++) {
                const x = bx + col * cs;
                const y = by + row * cs;
                if (board[row][col] === 1) {
                    const col_ = colorGrid[row][col] || 0x448aff;
                    gfx.fillStyle(col_);
                    gfx.fillRoundedRect(x + pad, y + pad, cs - pad * 2, cs - pad * 2, r);
                    gfx.fillStyle(Phaser.Display.Color.IntegerToColor(col_).lighten(30).color);
                    gfx.fillRoundedRect(x + pad + 2, y + pad + 2, cs - pad * 2 - 4, Math.floor(cs * 0.12), 2);
                } else {
                    gfx.fillStyle(0x181830);
                    gfx.fillRoundedRect(x + pad, y + pad, cs - pad * 2, cs - pad * 2, r);
                }
            }
        }

        if (finished) {
            gfx.fillStyle(0x000000, 0.52);
            gfx.fillRoundedRect(bx - 4, by - 4, bp + 8, bp + 8, 6);
        }
    }

    buildStatsPanel(W, H, agent) {
        const { bx, by, cs, bp } = agent;
        const panelX = bx + bp + Math.floor(cs * 0.9);
        const edgeX  = W - Math.floor(W * 0.025);
        const panelW = edgeX - panelX;

        if (panelW < 80) return;

        const fs     = Math.floor(H * 0.021);
        const lineH  = Math.floor(H * 0.052);
        const pad    = Math.floor(H * 0.016);
        const panelH = fs * 1.4 + lineH * 4 + pad * 2.5;
        const panelY = by + Math.floor((bp - panelH) / 2);

        const bg = this.add.graphics();
        bg.fillStyle(0x06091a, 0.88);
        bg.fillRoundedRect(panelX, panelY, panelW, panelH, 8);
        bg.lineStyle(1, 0x0d2244);
        bg.strokeRoundedRect(panelX, panelY, panelW, panelH, 8);

        const cx = panelX + panelW / 2;
        let y = panelY + pad;

        this.add.text(cx, y, 'Live Stats', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(fs * 1.05) + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
        }).setOrigin(0.5, 0);
        y += Math.floor(fs * 1.4) + pad * 0.5;

        const statDefs = [
            { label: 'Pieces',     key: 'piecesPlaced', fmt: v => String(v) },
            { label: 'Lines',      key: 'linesCleared', fmt: v => String(v) },
            { label: 'Best Combo', key: 'bestCombo',    fmt: v => '×' + v   },
            { label: 'Avg/Piece',  key: 'avgScore',     fmt: v => v.toFixed(1) },
        ];

        agent.statTxts = {};
        const lx = panelX + Math.floor(panelW * 0.10);
        const rx = panelX + panelW - Math.floor(panelW * 0.10);

        statDefs.forEach(def => {
            this.add.text(lx, y + lineH * 0.15, def.label, {
                fontFamily: 'Orbitron, Arial',
                fontSize: Math.floor(fs * 0.70) + 'px',
                color: '#2a5a7a',
            }).setOrigin(0, 0);

            const valTxt = this.add.text(rx, y + lineH * 0.15, '0', {
                fontFamily: 'Orbitron, Arial',
                fontSize: Math.floor(fs * 0.82) + 'px',
                fontStyle: 'bold',
                color: '#c8e8ff',
            }).setOrigin(1, 0);

            agent.statTxts[def.key] = { txt: valTxt, fmt: def.fmt };
            y += lineH;
        });
    }

    refreshStatsTxts(agent) {
        if (!agent.statTxts || !agent.state) return;
        const { stats, state } = agent;
        const avg = stats.piecesPlaced > 0 ? state.score / stats.piecesPlaced : 0;
        const vals = {
            piecesPlaced: stats.piecesPlaced,
            linesCleared: stats.linesCleared,
            bestCombo:    stats.bestCombo,
            avgScore:     avg,
        };
        Object.entries(agent.statTxts).forEach(([key, { txt, fmt }]) => {
            txt.setText(fmt(vals[key]));
        });
    }

    buildSpeedControls(W, H) {
        const presets = [0.5, 1, 2, 5, 10];
        const btnH = Math.floor(H * 0.040);
        const btnW = Math.floor(H * 0.064);
        const gap  = Math.floor(H * 0.010);
        const fs   = Math.floor(H * 0.019);
        const R    = Math.floor(btnH * 0.22);
        const totalW  = presets.length * btnW + (presets.length - 1) * gap;
        const startX  = W - Math.floor(W * 0.025) - totalW;
        const y       = Math.floor(H * 0.040);

        this._speedGfx  = [];
        this._speedTxts = [];

        presets.forEach((spd, i) => {
            const bx  = startX + i * (btnW + gap) + btnW / 2;
            const gfx = this.add.graphics();
            const txt = this.add.text(bx, y, spd + '×', {
                fontFamily: 'Orbitron, Arial',
                fontSize: fs + 'px',
                fontStyle: 'bold',
            }).setOrigin(0.5);

            this._speedGfx.push({ gfx, bx, y, btnW, btnH, R, spd });
            this._speedTxts.push(txt);

            const zone = this.add.zone(bx - btnW / 2, y - btnH / 2, btnW, btnH)
                .setOrigin(0, 0).setInteractive({ useHandCursor: true });
            zone.on('pointerdown', () => this.setSpeed(spd));
        });

        this.refreshSpeedControls();
    }

    refreshSpeedControls() {
        if (!this._speedGfx) return;
        this._speedGfx.forEach(({ gfx, bx, y, btnW, btnH, R, spd }, i) => {
            const active = spd === this.speed;
            gfx.clear();
            gfx.fillStyle(active ? 0x0a2060 : 0x06091a);
            gfx.fillRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
            gfx.lineStyle(1, active ? 0x00d4ff : 0x0d2244);
            gfx.strokeRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
            this._speedTxts[i].setStyle({ color: active ? '#00d4ff' : '#2a4a6a' });
        });
    }

    setSpeed(spd) {
        this.speed = spd;
        this.refreshSpeedControls();
        this.agents.forEach(agent => {
            if (agent.ws && agent.ws.readyState === WebSocket.OPEN) {
                agent.ws.send(JSON.stringify({ type: 'set_speed', speed: spd }));
            }
        });
    }

    buildMenuButton(H) {
        const btn = this.add.text(16, 16, '← Menu', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.022) + 'px',
            color: '#1a3a5a',
        }).setInteractive({ useHandCursor: true });
        btn.on('pointerover', () => btn.setStyle({ color: '#00d4ff' }));
        btn.on('pointerout',  () => btn.setStyle({ color: '#1a3a5a' }));
        btn.on('pointerdown', () => {
            try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
            this.agents.forEach(a => { if (a.ws) { a.ws.onmessage = null; a.ws.close(); } });
            this.scene.start('MenuScene');
        });
    }

    checkAllDone() {
        if (this.allDone) return;
        if (!this.agents.every(a => a.finished)) return;
        this.allDone = true;
        this.time.delayedCall(600, () => {
            this.agents.forEach(a => { if (a.ws) { a.ws.onmessage = null; a.ws.close(); } });
            this.scene.start('ComparisonScene', {
                results: this.agents.map(a => ({
                    agent:        a.id,
                    label:        a.label,
                    score:        a.finalScore,
                    piecesPlaced: a.stats.piecesPlaced,
                    linesCleared: a.stats.linesCleared,
                    bestCombo:    a.stats.bestCombo,
                })),
                agentIds: this.agentIds,
                speed:    this.speed,
                seed:     this.seed,
            });
        });
    }

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => {
            this.agents.forEach(a => { if (a.ws) { a.ws.onmessage = null; a.ws.close(); } });
            this.scene.restart({
                agents: this.agentIds,
                speed:  this.speed,
                seed:   Math.floor(Math.random() * 999999),
            });
        }, 200);
    }
}
