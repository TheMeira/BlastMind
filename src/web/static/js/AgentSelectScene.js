const AGENT_ROSTER = [
    { id: 'random', label: 'Random',     available: true  },
    { id: 'greedy', label: 'Greedy',     available: true  },
    { id: 'beam',   label: 'Beam Search',available: true  },
    { id: 'dqn',    label: 'DQN',        available: false },
    { id: 'mcts',   label: 'MCTS',       available: false },
];

class AgentSelectScene extends Phaser.Scene {
    constructor() {
        super({ key: 'AgentSelectScene' });
    }

    init(data) {
        this.fromScene = (data && data.from) || 'MenuScene';
        this.selected  = new Set(data && data.selected ? data.selected : ['random']);
        this.speed     = (data && data.speed) || 1.0;
    }

    preload() {
        if (!this.cache.audio.exists('click')) {
            this.load.audio('click', '/static/audio/click.mp3');
        }
    }

    create() {
        this.cameras.main.setBackgroundColor('#070714');
        const W = this.scale.width;
        const H = this.scale.height;
        const cx = W / 2;

        this.add.text(cx, H * 0.09, 'Watch AI', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(Math.min(H * 0.08, 70)) + 'px',
            fontStyle: 'bold',
            color: '#00d4ff',
            shadow: { offsetX: 0, offsetY: 0, color: '#00aaff', blur: 40, fill: true },
        }).setOrigin(0.5);

        this.add.text(cx, H * 0.21, 'Select one or more agents', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.024) + 'px',
            color: '#2a4a6a',
        }).setOrigin(0.5);

        this.buildAgentButtons(cx, H);
        this.buildSpeedSelector(cx, H);
        this.buildWatchButton(cx, H);

        this.makeButton(cx, H * 0.90, '← Back', H, () => {
            this.scene.start(this.fromScene);
        });

        this.scale.on('resize', this._onResize, this);
        this.events.once('shutdown', () => this.scale.off('resize', this._onResize, this));
    }

    buildAgentButtons(cx, H) {
        const btnW = Math.floor(Math.min(H * 0.22, 190));
        const btnH = Math.floor(H * 0.082);
        const gap  = Math.floor(H * 0.022);
        const totalW = AGENT_ROSTER.length * btnW + (AGENT_ROSTER.length - 1) * gap;
        const startX = cx - totalW / 2;
        const y = H * 0.38;
        const R  = Math.floor(btnH * 0.22);
        const fs = Math.floor(H * 0.024);

        this._agentGfx  = [];
        this._agentTxts = [];

        AGENT_ROSTER.forEach((agent, i) => {
            const bx  = startX + i * (btnW + gap) + btnW / 2;
            const gfx = this.add.graphics();
            const txt = this.add.text(bx, y, agent.label, {
                fontFamily: 'Orbitron, Arial',
                fontSize: fs + 'px',
                fontStyle: 'bold',
            }).setOrigin(0.5);

            this._agentGfx.push({ gfx, bx, y, btnW, btnH, R, agent });
            this._agentTxts.push(txt);

            if (!agent.available) {
                this.add.text(bx, y + Math.floor(btnH * 0.62), 'coming soon', {
                    fontFamily: 'Orbitron, Arial',
                    fontSize: Math.floor(fs * 0.58) + 'px',
                    color: '#1a3060',
                }).setOrigin(0.5, 0);
            } else {
                const zone = this.add.zone(bx - btnW / 2, y - btnH / 2, btnW, btnH)
                    .setOrigin(0, 0).setInteractive({ useHandCursor: true });
                zone.on('pointerdown', () => {
                    try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
                    if (this.selected.has(agent.id) && this.selected.size > 1) {
                        this.selected.delete(agent.id);
                    } else {
                        this.selected.add(agent.id);
                    }
                    this.refreshAgentButtons();
                });
            }
        });

        this.refreshAgentButtons();
    }

    refreshAgentButtons() {
        this._agentGfx.forEach(({ gfx, bx, y, btnW, btnH, R, agent }, i) => {
            const on = this.selected.has(agent.id);
            gfx.clear();
            if (!agent.available) {
                gfx.fillStyle(0x06091a);
                gfx.fillRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
                gfx.lineStyle(1, 0x0d1a2a);
                gfx.strokeRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
                this._agentTxts[i].setStyle({ color: '#1a3050' });
            } else if (on) {
                gfx.fillStyle(0x0a2060);
                gfx.fillRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
                gfx.lineStyle(2, 0x00d4ff);
                gfx.strokeRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
                this._agentTxts[i].setStyle({ color: '#00d4ff' });
            } else {
                gfx.fillStyle(0x060d28);
                gfx.fillRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
                gfx.lineStyle(2, 0x1a4888);
                gfx.strokeRoundedRect(bx - btnW / 2, y - btnH / 2, btnW, btnH, R);
                this._agentTxts[i].setStyle({ color: '#c8e8ff' });
            }
        });
    }

    buildSpeedSelector(cx, H) {
        const presets = [0.5, 1, 2, 5, 10];
        const btnW = Math.floor(H * 0.072);
        const btnH = Math.floor(H * 0.050);
        const gap  = Math.floor(H * 0.014);
        const fs   = Math.floor(H * 0.022);
        const R    = Math.floor(btnH * 0.22);
        const totalW = presets.length * btnW + (presets.length - 1) * gap;
        const startX = cx - totalW / 2;
        const y = H * 0.56;

        this.add.text(cx, y - Math.floor(H * 0.055), 'Speed', {
            fontFamily: 'Orbitron, Arial',
            fontSize: Math.floor(H * 0.024) + 'px',
            color: '#2a4a6a',
        }).setOrigin(0.5);

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
            zone.on('pointerdown', () => {
                try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
                this.speed = spd;
                this.refreshSpeedButtons();
            });
        });

        this.refreshSpeedButtons();
    }

    refreshSpeedButtons() {
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

    buildWatchButton(cx, H) {
        const W  = Math.floor(H * 0.42);
        const Ht = Math.floor(H * 0.082);
        const R  = Math.floor(Ht * 0.22);
        const fs = Math.floor(H * 0.030);
        const y  = H * 0.74;

        const gfx = this.add.graphics();
        const txt = this.add.text(cx, y, 'Watch', {
            fontFamily: 'Orbitron, Arial',
            fontSize: fs + 'px',
            fontStyle: 'bold',
            color: '#c8e8ff',
        }).setOrigin(0.5);

        this._watchGfx = gfx;
        this._watchTxt = txt;
        this._watchBtnParams = { cx, y, W, Ht, R };

        this.drawBtn(gfx, cx, y, W, Ht, R, false);

        const zone = this.add.zone(cx - W / 2, y - Ht / 2, W, Ht)
            .setOrigin(0, 0).setInteractive({ useHandCursor: true });
        zone.on('pointerover', () => { this.drawBtn(gfx, cx, y, W, Ht, R, true); txt.setStyle({ color: '#00d4ff' }); });
        zone.on('pointerout',  () => { this.drawBtn(gfx, cx, y, W, Ht, R, false); txt.setStyle({ color: '#c8e8ff' }); });
        zone.on('pointerdown', () => {
            try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
            const seed = Math.floor(Math.random() * 999999);
            this.scene.start('MultiAgentScene', {
                agents: Array.from(this.selected),
                speed:  this.speed,
                seed,
            });
        });
    }

    makeButton(x, y, label, H, callback) {
        const W  = Math.floor(H * 0.38);
        const Ht = Math.floor(H * 0.072);
        const R  = Math.floor(Ht * 0.22);
        const fs = Math.floor(H * 0.026);
        const gfx = this.add.graphics();
        this.drawBtn(gfx, x, y, W, Ht, R, false);
        const txt = this.add.text(x, y, label, {
            fontFamily: 'Orbitron, Arial', fontSize: fs + 'px', fontStyle: 'bold', color: '#c8e8ff',
        }).setOrigin(0.5);
        const zone = this.add.zone(x - W / 2, y - Ht / 2, W, Ht).setOrigin(0, 0).setInteractive({ useHandCursor: true });
        zone.on('pointerover', () => { this.drawBtn(gfx, x, y, W, Ht, R, true); txt.setStyle({ color: '#00d4ff' }); });
        zone.on('pointerout',  () => { this.drawBtn(gfx, x, y, W, Ht, R, false); txt.setStyle({ color: '#c8e8ff' }); });
        zone.on('pointerdown', () => {
            try { this.sound.play('click', { volume: getSFXVolume() }); } catch (e) {}
            this.time.delayedCall(140, callback);
        });
    }

    drawBtn(gfx, x, y, W, H, R, hover) {
        gfx.clear();
        gfx.fillStyle(hover ? 0x0a2060 : 0x060d28);
        gfx.fillRoundedRect(x - W / 2, y - H / 2, W, H, R);
        gfx.lineStyle(2, hover ? 0x00d4ff : 0x1a4888);
        gfx.strokeRoundedRect(x - W / 2, y - H / 2, W, H, R);
    }

    _onResize() {
        clearTimeout(this._resizeTimer);
        this._resizeTimer = setTimeout(() => {
            this.scene.restart({ from: this.fromScene, selected: Array.from(this.selected), speed: this.speed });
        }, 200);
    }
}
